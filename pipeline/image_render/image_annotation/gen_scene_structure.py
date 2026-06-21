import json
import numpy as np
from PIL import Image
import cv2
import os
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import traceback
from tqdm import tqdm
import time

'''
# for an instance: 1940Office/l000_r001
Image: /path/to/Taxonomy/image_render/Test_Everything_Data/jiawei/1940Office/l000_r001/lit.png
All object annotations: /path/to/Taxonomy/image_render/Test_Everything_Data/jiawei/1940Office/l000_r001/object_annots.json
In the frame objects annotations (segmentation color rgb): /path/to/Taxonomy/image_render/Test_Everything_Data/jiawei/1940Office/l000_r001/seenable_obj_dict.json
Segmentation mask: /path/to/Taxonomy/image_render/Test_Everything_Data/jiawei/1940Office/l000_r001/seg.png
Depth: /path/to/Taxonomy/image_render/Test_Everything_Data/jiawei/1940Office/l000_r001/depth.npy (skip for now)
'''

# Global cache for mappings
_CLASSES_TO_INSTANCES = None
_BACKGROUND_OBJECTS = None
_INSTANCE_TO_CLASS = None


def load_mappings():
    """Load the classes to instances and background objects mappings."""
    global _CLASSES_TO_INSTANCES, _BACKGROUND_OBJECTS, _INSTANCE_TO_CLASS
    
    if _CLASSES_TO_INSTANCES is None:
        # Use fixed path to SimulationMetadata/objects
        metadata_dir = Path('/path/to/Taxonomy/Data/SimulationMetadata/objects')
                
        # Load classes to instances mapping
        classes_path = metadata_dir / 'objects_list' / 'new_all_object_list.json'
        with open(classes_path, 'r') as f:
            _CLASSES_TO_INSTANCES = json.load(f)
        
        # Build reverse mapping: instance name -> class
        _INSTANCE_TO_CLASS = {}
        for category, instances in _CLASSES_TO_INSTANCES.items():
            for instance in instances:
                obj_name = instance['object name']
                _INSTANCE_TO_CLASS[obj_name] = category
        
        # Load background objects list
        bg_path = metadata_dir / 'background_objects.json'
        with open(bg_path, 'r') as f:
            bg_data = json.load(f)
            _BACKGROUND_OBJECTS = set(bg_data['Background_Objects'])
    
    return _INSTANCE_TO_CLASS, _BACKGROUND_OBJECTS


def extract_category_from_object_id(object_id):
    """
    Extract category from object_id using the classes_to_instances mapping.
    Falls back to parsing the object_id if not found in mapping.
    
    Args:
        object_id: Object ID string (e.g., 'SM_Floor_4x4m46')
    
    Returns:
        category: Category name (e.g., 'floor')
    """
    instance_to_class, _ = load_mappings()
    
    # Try direct lookup first
    if object_id in instance_to_class:
        return instance_to_class[object_id]

    
    # Try with numbers removed
    import re
    # Remove trailing numbers and underscores
    # base_name = re.sub(r'[_\-]\d+$', '', object_id)
    base_name = '_'.join(object_id.split('_')[:-1])
    if base_name in instance_to_class:
        return instance_to_class[base_name]

    return None
    
def is_background_object(category):
    """
    Check if a category is a background object.
    
    Args:
        category: Category name
    
    Returns:
        bool: True if background object, False if foreground
    """
    _, background_objects = load_mappings()
    return category.lower() in background_objects


def check_has_foreground(data_dir):
    """
    Check if a scene has any foreground objects before processing.
    
    Args:
        data_dir: Path to the directory containing the data files
    
    Returns:
        bool: True if has foreground objects, False otherwise
    """
    data_dir = Path(data_dir)
    
    # Check if seenable_obj_dict.json exists
    seenable_obj_path = data_dir / 'seenable_obj_dict.json'
    if not seenable_obj_path.exists():
        return False
    
    # Load the seenable objects dictionary
    with open(seenable_obj_path, 'r') as f:
        seenable_obj_dict = json.load(f)
    
    # Check if any object is foreground
    for object_id in seenable_obj_dict.keys():
        category = extract_category_from_object_id(object_id)
        if category is None:
            continue
        
        # If it's not a background object, it's foreground
        if not is_background_object(category):
            return True
    
    # No foreground objects found
    return False


def seg_to_bbox(data_dir, output_dir=None):
    """
    Convert segmentation mask to bounding boxes.
    
    Args:
        data_dir: Path to the directory containing the data files
                 (e.g., /path/to/1940Office/l000_r001)
        output_dir: Path to the output directory. If None, saves to data_dir
    """
    data_dir = Path(data_dir)
    
    # Set output directory
    if output_dir is None:
        output_base = data_dir
    else:
        output_base = Path(output_dir)
    
    # Load the segmentation mask
    seg_path = data_dir / 'seg.png'
    seg_img = np.array(Image.open(seg_path))
    
    # Convert RGBA to RGB if needed
    if seg_img.shape[-1] == 4:
        seg_img = seg_img[:, :, :3]
    
    # Load the seenable objects dictionary (RGB colors for each object)
    seenable_obj_path = data_dir / 'seenable_obj_dict.json'
    with open(seenable_obj_path, 'r') as f:
        seenable_obj_dict = json.load(f)
    
    # Load camera annotations
    camera_annots_path = data_dir / 'camera_annots.json'
    camera_info = {}
    if camera_annots_path.exists():
        with open(camera_annots_path, 'r') as f:
            camera_data = json.load(f)
            if 'outputs' in camera_data:
                camera_info = camera_data['outputs']
            elif camera_data.get('status') == 'ok':
                camera_info = camera_data.get('outputs', {})
    
    # Load object annotations
    object_annots_path = data_dir / 'object_annots.json'
    with open(object_annots_path, 'r') as f:
        object_annots_data = json.load(f)
    
    # Build a dictionary for quick lookup: object_id -> 3D info
    object_3d_info = {}
    if 'outputs' in object_annots_data:
        for obj in object_annots_data['outputs']:
            obj_id = obj.get('object_id')
            if obj_id:
                object_3d_info[obj_id] = {
                    'aabb': obj.get('aabb', {}),
                    'obb': obj.get('obb', {}),
                    'location': obj.get('location', [0, 0, 0]),
                    'rotation': obj.get('rotation', [0, 0, 0]),
                    'scale': obj.get('scale', [1, 1, 1])
                }
    
    # Create annotations lists
    all_annotations = []
    background_annotations = []
    foreground_annotations = []
    
    # Process each object in the frame
    for object_id, rgb_color in seenable_obj_dict.items():
        
        # Extract category from object_id
        category = extract_category_from_object_id(object_id)
        if category is None:
            continue
        
        # Determine if background or foreground
        is_background = is_background_object(category)
        object_type = 'background' if is_background else 'foreground'
        
        # Find all pixels matching this RGB color
        rgb_color = np.array(rgb_color, dtype=seg_img.dtype)
        mask = np.all(seg_img == rgb_color, axis=-1)
        
        # Skip if no pixels found
        if not mask.any():
            continue
        
        # Get bounding box coordinates
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
            
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        

        
        # Get 3D information for this object
        bbox_3d = object_3d_info.get(object_id, {})
        
        # Create annotation entry
        annotation = {
            'object_id': object_id,
            'bbox_2d': [int(x_min), int(y_min), int(x_max), int(y_max)],
            'bbox_3d': {
                'aabb': bbox_3d.get('aabb', {}),
                'obb': bbox_3d.get('obb', {}),
                'location': bbox_3d.get('location', [0, 0, 0]),
                'rotation': bbox_3d.get('rotation', [0, 0, 0]),
                'scale': bbox_3d.get('scale', [1, 1, 1])
            },
            'category': category,
            'object_type': object_type,
            'color': rgb_color.tolist()  # Store segmentation color as [R, G, B]
        }
        
        # Add to appropriate lists
        all_annotations.append(annotation)
        if is_background:
            background_annotations.append(annotation)
        else:
            foreground_annotations.append(annotation)
    
    # Check if there are any foreground objects
    if len(foreground_annotations) == 0:
        print(f"Skipping - no foreground objects found")
        return None, seg_img, None
    
    # Create output directory if needed
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Group background objects by category
    background_by_category = {}
    for annot in background_annotations:
        category = annot['category']
        if category not in background_by_category:
            background_by_category[category] = []
        background_by_category[category].append({
            'object_id': annot['object_id'],
            'bbox_2d': annot['bbox_2d'],
            'bbox_3d': annot['bbox_3d'],
            'color': annot['color']  # Save the segmentation color
        })
    
    # Group foreground objects by category
    foreground_by_category = {}
    for annot in foreground_annotations:
        category = annot['category']
        if category not in foreground_by_category:
            foreground_by_category[category] = []
        foreground_by_category[category].append({
            'object_id': annot['object_id'],
            'bbox_2d': annot['bbox_2d'],
            'bbox_3d': annot['bbox_3d'],
            'color': annot['color']  # Save the segmentation color
        })
    
    # Extract scene metadata
    lit_path = data_dir / 'lit.png'
    scene_parts = data_dir.parts
    # Try to find scene name and view ID from path
    # Typical path: .../simulationImage/user/SceneName/view_id/
    scene_name = scene_parts[-2] if len(scene_parts) >= 2 else "unknown"
    view_id = scene_parts[-1] if len(scene_parts) >= 1 else "unknown"
    
    # Create split annotations with new structure
    split_annotations = {
        'scene_name': scene_name,
        'view_id': view_id,
        'image_path': str(lit_path),
        'camera': camera_info,
        'background': background_by_category,
        'foreground': foreground_by_category
    }
    split_output_path = output_base / 'scene_annotations_split.json'
    with open(split_output_path, 'w') as f:
        json.dump(split_annotations, f, indent=2)
    
    print(f"Saved {len(all_annotations)} annotations")
    print(f"  - Background: {len(background_annotations)} objects in {len(background_by_category)} categories")
    print(f"  - Foreground: {len(foreground_annotations)} objects in {len(foreground_by_category)} categories")
    print(f"Saved to {split_output_path}")
    
    return all_annotations, seg_img, output_base, camera_info


def project_3d_to_2d(point_3d, c2w_matrix, camera_location, fxfycxcy, width, height):
    """
    Project a 3D point to 2D image coordinates (Unreal Engine coordinate system).
    
    Args:
        point_3d: [x, y, z] in Unreal world coordinates
        c2w_matrix: camera-to-world matrix (4x4)
        camera_location: camera location [x, y, z]
        fxfycxcy: [fx, fy, cx, cy] camera intrinsics
        width, height: image dimensions
    
    Returns:
        [u, v] in image coordinates, or None if behind camera
    """
    import numpy as np
    
    # Unreal to OpenCV coordinate conversion
    UE2CV = np.array([
        [-1,  0, 0],
        [0,  1, 0],
        [0,  0, 1]
    ])
    
    # Extract rotation from c2w
    c2w = np.array(c2w_matrix).T
    R = c2w[:3, :3]
    t = np.array(camera_location)
    
    # Flip X coordinate (Unreal convention)
    point = np.array(point_3d)
    
    # World to camera: (R.T @ (point - t))
    cam_pt = (R.T @ (point - t))
    
    # Unreal coordinates: X=forward, Y=right, Z=up
    x, y, z = cam_pt
    
    # Check if behind camera
    if x <= 1e-5:
        return None
    
    # Project to image plane
    fx, fy, cx, cy = fxfycxcy
    u = fx * (y / x) + cx
    v = -fy * (z / x) + cy
    
    # Check if point is within image bounds (with margin)
    if -width <= u < width*2 and -height <= v < height*2:
        return [int(u), int(v)]
    return None


def get_obb_corners(obb):
    """
    Get the 8 corners of an oriented bounding box.
    
    Args:
        obb: dict with 'center', 'extent', 'rotation'
    
    Returns:
        List of 8 corner points in world coordinates
    """
    import numpy as np
    
    center = np.array(obb.get('center', [0, 0, 0]))
    extent = np.array(obb.get('extent', [1, 1, 1]))
    rotation = np.array(obb.get('rotation', [0, 0, 0]))  # [pitch, yaw, roll] in degrees
    
    # Convert rotation to radians
    pitch, yaw, roll = np.radians(rotation)
    
    # Create rotation matrix (ZYX order)
    # Yaw (Z), Pitch (Y), Roll (X)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    
    R_x = np.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]])
    R_y = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
    R_z = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
    
    R = R_z @ R_y @ R_x
    
    # Define the 8 corners of a unit box
    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]   # Top face
    ])
    
    # Scale by extent
    corners = corners * extent
    
    # Rotate
    corners = (R @ corners.T).T
    
    # Translate to center
    corners = corners + center
    
    return corners.tolist()


def visualize_segmentation_and_bbox(data_dir, annotations=None, seg_img=None, output_dir=None, camera_info=None, skip_3d=False):
    """
    Visualize 2D segmentation and bounding boxes.
    
    Args:
        data_dir: Path to the directory containing the data files
        annotations: Optional pre-loaded annotations
        seg_img: Optional pre-loaded segmentation image
        output_dir: Path to the output directory. If None, uses data_dir
        camera_info: Camera information for 3D bbox projection
    """
    data_dir = Path(data_dir)
    
    # Set output directory
    if output_dir is None:
        output_base = data_dir
    else:
        output_base = Path(output_dir)
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Load camera info if not provided
    if camera_info is None:
        camera_path = data_dir / 'camera_annots.json'
        if camera_path.exists():
            with open(camera_path, 'r') as f:
                camera_data = json.load(f)
                if camera_data.get('status') == 'ok':
                    camera_info = camera_data.get('outputs', {})
    
    # Load the original image
    lit_path = data_dir / 'lit.png'
    lit_img = cv2.imread(str(lit_path))
    
    # Load segmentation if not provided
    if seg_img is None:
        seg_path = data_dir / 'seg.png'
        seg_img = np.array(Image.open(seg_path))
        # Convert RGBA to RGB if needed
        if seg_img.shape[-1] == 4:
            seg_img = seg_img[:, :, :3]
    
    # Load annotations if not provided
    if annotations is None:
        annot_path = output_base / 'scene_annotations_split.json'
        with open(annot_path, 'r') as f:
            split_data = json.load(f)
            # Reconstruct annotations from new structure
            annotations = []
            # Add background objects (now grouped by category)
            for category, objects in split_data['background'].items():
                for obj in objects:
                    annotations.append({
                        'object_id': obj['object_id'],
                        'bbox_2d': obj.get('bbox_2d', obj.get('bbox', [0, 0, 0, 0])),  # Support old format
                        'category': category,
                        'object_type': 'background',
                        'color': obj.get('color', [128, 128, 128])  # Default gray if not found
                    })
            # Add foreground objects (now grouped by category)
            for category, objects in split_data['foreground'].items():
                for obj in objects:
                    annotations.append({
                        'object_id': obj['object_id'],
                        'bbox_2d': obj.get('bbox_2d', obj.get('bbox', [0, 0, 0, 0])),  # Support old format
                        'category': category,
                        'object_type': 'foreground',
                        'color': obj.get('color', [0, 255, 0])  # Default green if not found
                    })
    
    # Create visualization with bounding boxes (all objects)
    bbox_img_all = lit_img.copy()
    
    # Draw bounding boxes
    for annot in annotations:
        bbox_2d = annot['bbox_2d']
        x_min, y_min, x_max, y_max = bbox_2d
        category = annot['category']
        
        # Use the segmentation color (RGB -> BGR for OpenCV)
        seg_color = annot.get('color', [0, 255, 0])  # Default to green if not found
        color = (seg_color[2], seg_color[1], seg_color[0])  # Convert RGB to BGR
        
        if category == 'background':
            color = (128, 128, 128)  # Default gray if not found
        
        # Draw rectangle
        cv2.rectangle(bbox_img_all, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Put category label
        label = f"{category}"
        font_scale = 1.2  # Increased from 0.5
        thickness = 4  # Increased from 1
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.rectangle(bbox_img_all, (x_min, y_min - label_size[1] - 8), 
                     (x_min + label_size[0] + 4, y_min), color, -1)
        cv2.putText(bbox_img_all, label, (x_min + 2, y_min - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Resize to 1/2 size
    height, width = bbox_img_all.shape[:2]
    new_height, new_width = height // 4, width // 4
    bbox_img_resized = cv2.resize(bbox_img_all, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Save only the all-objects visualization (resized to 1/2)
    bbox_vis_all_path = output_base / 'bbox_visualization_all.png'
    cv2.imwrite(str(bbox_vis_all_path), bbox_img_resized)
    
    print(f"Saved bbox visualization (all, {new_width}x{new_height}) to {bbox_vis_all_path}")
    
    # Create 3D bounding box visualization
    if not skip_3d and camera_info and camera_info.get('c2w') and camera_info.get('fxfycxcy') and camera_info.get('location'):
        bbox_img_3d = lit_img.copy()
        c2w = camera_info['c2w']
        camera_location = camera_info['location']
        fxfycxcy = camera_info['fxfycxcy']
        img_width = camera_info.get('width', lit_img.shape[1])
        img_height = camera_info.get('height', lit_img.shape[0])
        
        drawn_count = 0
        skipped_count = 0
        
        for annot in annotations:
            # Skip if no 3D bbox info
            if 'bbox_3d' not in annot or 'aabb' not in annot.get('bbox_3d', {}):
                skipped_count += 1
                continue
            
            aabb = annot['bbox_3d']['aabb']
            if not aabb.get('center') or not aabb.get('extent'):
                skipped_count += 1
                continue
            
            # Get 8 corners of the AABB (axis-aligned bounding box)
            center = np.array(aabb['center'])
            extent = np.array(aabb['extent'])
            corners_3d = []
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        corner = center + extent * np.array([dx, dy, dz])
                        corners_3d.append(corner.tolist())
            
            # Project all corners to 2D (even if outside image bounds)
            corners_2d = []
            for corner in corners_3d:
                proj = project_3d_to_2d(corner, c2w, camera_location, fxfycxcy, img_width * 2, img_height * 2)  # Expand bounds
                if proj:
                    corners_2d.append(proj)
                else:
                    corners_2d.append(None)
            
            # Draw 3D box if we have at least some visible corners
            if any(c is not None for c in corners_2d):
                # Use the segmentation color (brighter for 3D)
                seg_color = annot.get('color', [0, 255, 0])
                color = (seg_color[2], seg_color[1], seg_color[0])  # RGB to BGR
                
                if annot['category'] == 'background':
                    color = (128, 128, 128)
                # Draw edges of the 3D box
                # Corner order: [dx, dy, dz] where dx=-1/+1, dy=-1/+1, dz=-1/+1
                # 0:(-1,-1,-1), 1:(-1,-1,+1), 2:(-1,+1,-1), 3:(-1,+1,+1),
                # 4:(+1,-1,-1), 5:(+1,-1,+1), 6:(+1,+1,-1), 7:(+1,+1,+1)
                edges = [
                    (0, 1), (0, 2), (0, 4),  # From corner 0
                    (1, 3), (1, 5),          # From corner 1
                    (2, 3), (2, 6),          # From corner 2
                    (3, 7),                  # From corner 3
                    (4, 5), (4, 6),          # From corner 4
                    (5, 7),                  # From corner 5
                    (6, 7)                   # From corner 6
                ]
                
                for i, j in edges:
                    if corners_2d[i] is not None and corners_2d[j] is not None:
                        pt1 = tuple(corners_2d[i])
                        pt2 = tuple(corners_2d[j])
                        # Clip to image bounds
                        if (0 <= pt1[0] < img_width and 0 <= pt1[1] < img_height or
                            0 <= pt2[0] < img_width and 0 <= pt2[1] < img_height):
                            cv2.line(bbox_img_3d, pt1, pt2, color, 3)  # Thicker line
                
                drawn_count += 1
            else:
                skipped_count += 1
        
        # Resize and save 3D visualization
        bbox_img_3d_resized = cv2.resize(bbox_img_3d, (new_width, new_height), interpolation=cv2.INTER_AREA)
        bbox_vis_3d_path = output_base / 'bbox_3d_visualization.png'
        cv2.imwrite(str(bbox_vis_3d_path), bbox_img_3d_resized)
        print(f"Saved 3D bbox visualization ({new_width}x{new_height}) to {bbox_vis_3d_path}")
        print(f"  - Drew {drawn_count} 3D boxes, skipped {skipped_count}")


def process_single_scene(args):
    """
    Worker function to process a single scene (for parallel processing).
    
    Args:
        args: Tuple of (scene_dir, output_base, visualize, index, total)
    
    Returns:
        dict: Result with status and information
    """
    scene_dir, output_base, visualize, idx, total = args
    
    result = {
        'status': 'success',
        'scene_dir': scene_dir,
        'message': ''
    }
    
    try:
        # Check if required files exist
        seg_file = scene_dir / 'seg.png'
        if not seg_file.exists():
            result['status'] = 'skipped_no_seg'
            result['message'] = 'no seg.png'
            return result
        
        # Create output directory structure
        scene_folder = scene_dir.parent.name
        instance_folder = scene_dir.name
        scene_output_dir = output_base / scene_folder / instance_folder
        
        print(f"[{idx}/{total}] Processing {scene_folder}/{instance_folder}...")
        
        # check if scene_annotations_split.json already exists
        if (scene_output_dir / 'scene_annotations_split.json').exists():
            print(f"Skipping - {scene_output_dir / 'scene_annotations_split.json'} already exists")
            result['status'] = 'skipped_already_processed'
            result['message'] = f"{scene_folder}/{instance_folder}"
            print(f"  -> Skipped (already processed)")
            return result
        
        # Check for foreground objects BEFORE processing
        if not check_has_foreground(scene_dir):
            result['status'] = 'skipped_no_foreground'
            result['message'] = f"{scene_folder}/{instance_folder}"
            print(f"  -> Skipped (no foreground objects)")
            return result
        
        # Process the scene
        annotations, seg_img, actual_output, camera_info = seg_to_bbox(scene_dir, scene_output_dir)
        
        # Double check
        if annotations is None:
            result['status'] = 'skipped_no_foreground'
            result['message'] = f"{scene_folder}/{instance_folder}"
            print(f"  -> Skipped (no foreground objects after processing)")
            return result
        
        if visualize:
            visualize_segmentation_and_bbox(scene_dir, annotations, seg_img, scene_output_dir, camera_info, skip_3d=False)
        
        result['message'] = f"{scene_folder}/{instance_folder}"
        print(f"  -> Success")
        
    except Exception as e:
        scene_folder = scene_dir.parent.name
        instance_folder = scene_dir.name
        result['status'] = 'error'
        result['message'] = f"{scene_folder}/{instance_folder}: {str(e)}"
        print(f"  -> Error: {e}")
        traceback.print_exc()
    
    return result


def process_from_json(json_file, output_base_dir, visualize=False, num_workers=None, n_samples=None):
    """
    Process scene directories listed in a JSON file (e.g., image_quality_ratings.json).
    
    Args:
        json_file: Path to JSON file with image paths as keys
        output_base_dir: Base output directory
        visualize: Whether to generate visualizations
        num_workers: Number of parallel workers (default: CPU count)
    """
    output_base = Path(output_base_dir)
    
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract unique scene directories from image paths
    scene_dirs = []
    for image_path in data.keys():
        # Image path is like: .../scene_folder/instance_folder/lit.png
        # We need the parent directory (instance_folder)
        scene_dir = Path(image_path).parent
        if scene_dir not in scene_dirs:
            scene_dirs.append(scene_dir)
    if n_samples is not None:
        scene_dirs = scene_dirs[:n_samples]
    total_scenes = len(scene_dirs)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 16)  # Cap at 16 to avoid too many processes
    
    print(f"Found {total_scenes} scene directories to process from {json_file}")
    print(f"Output will be saved to: {output_base}")
    print(f"Using {num_workers} parallel workers")
    print(f"Processing {n_samples} samples" if n_samples is not None else "Processing all samples")
    print(f"{'='*60}\n")
    
    # Prepare arguments for parallel processing
    tasks = [(scene_dir, output_base, visualize, i+1, total_scenes) 
             for i, scene_dir in enumerate(scene_dirs)]
    
    # Process in parallel with progress bar
    success_count = 0
    error_count = 0
    skipped_no_seg = 0
    skipped_no_foreground = []
    skipped_already_processed = []
    error_list = []
    
    print("Starting to process scenes...")
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        # Use tqdm to show progress bar with time estimation
        with tqdm(total=total_scenes, desc="Scene Processing Progress", 
                 unit="scene", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            results = []
            # Use imap_unordered for real-time progress updates
            for result in pool.imap_unordered(process_single_scene, tasks):
                results.append(result)
                # Update counters
                if result['status'] == 'success':
                    success_count += 1
                elif result['status'] == 'skipped_no_seg':
                    skipped_no_seg += 1
                elif result['status'] == 'skipped_no_foreground':
                    skipped_no_foreground.append(result['message'])
                elif result['status'] == 'skipped_already_processed':
                    skipped_already_processed.append(result['message'])
                elif result['status'] == 'error':
                    error_list.append(result['message'])
                    error_count += 1
                
                # Update progress bar description with current stats and time
                elapsed_time = time.time() - start_time
                desc = f"Processing Scenes [Success:{success_count} Errors:{error_count} Skipped:{len(skipped_no_foreground)+skipped_no_seg}] Time:{elapsed_time:.1f}s"
             
                pbar.set_description(desc)
                pbar.update(1)
    
    # Save error lists
    if skipped_no_foreground or error_list:
        error_file = output_base / 'processing_errors.txt'
        with open(error_file, 'w') as f:
            f.write(f"Processing Report\n")
            f.write(f"{'='*60}\n\n")
            
            if skipped_no_foreground:
                f.write(f"Skipped (No Foreground Objects): {len(skipped_no_foreground)}\n")
                f.write(f"{'-'*60}\n")
                for path in skipped_no_foreground:
                    f.write(f"{path}\n")
                f.write(f"\n")
            
            if error_list:
                f.write(f"Errors: {len(error_list)}\n")
                f.write(f"{'-'*60}\n")
                for error in error_list:
                    f.write(f"{error}\n")
        
        print(f"\nError list saved to: {error_file}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Success: {success_count}")
    print(f"Skipped (no seg.png): {skipped_no_seg}")
    print(f"Skipped (no foreground): {len(skipped_no_foreground)}")
    print(f"Errors: {error_count}")
    print(f"{'='*60}")


def process_batch(root_dir, output_base_dir, visualize=False, num_workers=None, n_samples=None):
    """
    Process all scene directories under a root directory in parallel.
    
    Args:
        root_dir: Root directory containing scene directories
        output_base_dir: Base output directory
        visualize: Whether to generate visualizations
        num_workers: Number of parallel workers (default: auto)
    """
    root_path = Path(root_dir)
    output_base = Path(output_base_dir)
    
    print(f"{'='*60}")
    print(f"Batch Processing Mode (Parallel)")
    print(f"{'='*60}")
    print(f"Scanning for scenes in: {root_path}")
    
    previous_error_file = output_base / 'processing_errors.txt'
    if previous_error_file.exists():
        with open(previous_error_file, 'r') as f:
            previous_error_list = f.readlines()
        previous_error_list = [line.strip().split(':')[0] for line in previous_error_list if line.strip()]
        previous_error_list = list(set(previous_error_list))
    
    # Find all directories containing seg.png
    scene_dirs = []
    for seg_file in root_path.rglob('lit.png'):
        # import ipdb; ipdb.set_trace()
        # # if output directory already exists, skip
        name = str(seg_file.parent).split('/')
        possible_error = f"{name[-2]}/{name[-1]}"
        if possible_error in previous_error_list:
            scene_dirs.append(seg_file.parent)
        
    print(f"Found {len(scene_dirs)} scene directories to process")
    total_scenes = len(scene_dirs)
    if n_samples is not None:
        scene_dirs = scene_dirs[:n_samples]
        total_scenes = len(scene_dirs)
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Default to 8 workers
    
    print(f"Found {total_scenes} scene directories to process")
    print(f"Output will be saved to: {output_base}")
    print(f"Using {num_workers} parallel workers")
    print(f"Processing {n_samples} samples" if n_samples is not None else "Processing all samples")
    print(f"{'='*60}\n")
    
    # Prepare arguments for parallel processing
    tasks = [(scene_dir, output_base, visualize, i+1, total_scenes) 
             for i, scene_dir in enumerate(scene_dirs)]
    
    # Process in parallel with progress bar
    success_count = 0
    error_count = 0
    skipped_no_seg = 0
    skipped_no_foreground = []
    skipped_already_processed = []
    error_list = []
    
    print("Starting to process scenes...")
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        # Use tqdm to show progress bar with time estimation
        with tqdm(total=total_scenes, desc="Batch Processing Progress", 
                 unit="scene", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            results = []
            # Use imap_unordered for real-time progress updates
            for result in pool.imap_unordered(process_single_scene, tasks):
                results.append(result)
                # Update counters
                if result['status'] == 'success':
                    success_count += 1
                elif result['status'] == 'skipped_no_seg':
                    skipped_no_seg += 1
                elif result['status'] == 'skipped_no_foreground':
                    skipped_no_foreground.append(result['message'])
                elif result['status'] == 'skipped_already_processed':
                    skipped_already_processed.append(result['message'])
                elif result['status'] == 'error':
                    error_list.append(result['message'])
                    error_count += 1
                
                # Update progress bar description with current stats and time
                elapsed_time = time.time() - start_time
                desc = f"Batch Processing [Success:{success_count} Errors:{error_count} Skipped:{len(skipped_no_foreground)+skipped_no_seg}] Time:{elapsed_time:.1f}s"
                pbar.set_description(desc)
                pbar.update(1)
    
    # Save error lists
    if skipped_no_foreground or error_list:
        error_file = output_base / 'processing_errors.txt'
        with open(error_file, 'w') as f:
            f.write(f"Processing Report\n")
            f.write(f"{'='*60}\n\n")
            
            if skipped_no_foreground:
                f.write(f"Skipped (No Foreground Objects): {len(skipped_no_foreground)}\n")
                f.write(f"{'-'*60}\n")
                for path in skipped_no_foreground:
                    f.write(f"{path}\n")
                f.write(f"\n")
            
            if error_list:
                f.write(f"Errors: {len(error_list)}\n")
                f.write(f"{'-'*60}\n")
                for error in error_list:
                    f.write(f"{error}\n")
        
        print(f"\nError list saved to: {error_file}")
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Success: {success_count}")
    print(f"Skipped (no seg.png): {skipped_no_seg}")
    print(f"Skipped (no foreground): {len(skipped_no_foreground)}")
    print(f"Skipped (already processed): {len(skipped_already_processed)}")
    print(f"Errors: {error_count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert segmentation masks to bounding boxes and visualize'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        nargs='?',
        default='/path/to/Taxonomy/Data/simulationImage',
        help='Path to the data directory (e.g., /path/to/1940Office/l000_r001) or base directory for batch processing'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations',
        help='Output directory for annotations and visualizations'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization images'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all scene directories recursively under data_dir'
    )
    parser.add_argument(
        '--from_json',
        type=str,
        default='/path/to/Taxonomy/scripts/image_render/image_process_quality_filter/image_quality_ratings.json',
        help='Process images from JSON file (e.g., image_quality_ratings.json)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of samples to process (default: None, process all)'
    )
    parser.add_argument(
        '--use_json',
        action='store_true',
        help='Use the JSON file specified by --from_json instead of data_dir'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4, auto for batch mode)'
    )
    parser.add_argument(
        '--skip_3d',
        action='store_true',
        help='Skip 3D bbox visualization (projection may be inaccurate)'
    )
    
    args = parser.parse_args()
    
    if args.use_json:
        # Process from JSON file
        print(f"Processing images from JSON file: {args.from_json}")
        process_from_json(args.from_json, args.output_dir, args.visualize, args.num_workers, args.n_samples)
    elif args.batch:
        # Process batch
        if not args.data_dir:
            parser.error("data_dir is required when using --batch")
        process_batch(args.data_dir, args.output_dir, args.visualize, args.num_workers, args.n_samples)
    else:
        # Process single directory
        if not args.data_dir:
            parser.error("data_dir is required (or use --use_json)")
        
        data_path = Path(args.data_dir)
        output_base = Path(args.output_dir)
        
        # Extract scene name and view ID from path
        # Path structure: .../user/scene/view_id/
        # Output structure: output_base/scene/view_id/
        scene_name = data_path.parent.name  # Get scene name (e.g., 1940Office)
        view_id = data_path.name             # Get view ID (e.g., l000_r001)
        scene_output_dir = output_base / scene_name / view_id
        
        expected_output_json = scene_output_dir / 'scene_annotations_split.json'
        # if expected_output_json.exists():
        #     print(f"Skipping - {expected_output_json} already exists")
        #     print("Done!")
        #     return
        
        # Check for foreground objects BEFORE processing
        if not check_has_foreground(args.data_dir):
            print(f"Skipping - no foreground objects found in {scene_name}/{view_id}")
            print("Done!")
            return
        
        annotations, seg_img, actual_output, camera_info = seg_to_bbox(args.data_dir, scene_output_dir)
        
        # Visualize if requested
        if args.visualize and annotations is not None:
            visualize_segmentation_and_bbox(args.data_dir, annotations, seg_img, scene_output_dir, camera_info, skip_3d=args.skip_3d)
        
        print("Done!")


if __name__ == '__main__':
    main()


