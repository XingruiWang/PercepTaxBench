import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available, bbox coverage filtering will be skipped")

class AnnotationProcessingUtils:
    """Utility class for processing annotation files"""
    
    # Human person mappings - normalize all people-related detections to 'person'
    # Includes: gender/age variants, role-specific detections (athlete, spectator, etc.)
    # Rationale: Most questions (material, affordance, spatial, capability) don't need role specificity
    # Normalizing ensures consistency and reduces confusion from multiple "person" variants
    PERSON_NORMALIZATION = {
        # Gender/age variants
        'woman': 'person',
        'man': 'person',
        'child': 'person',
        'boy': 'person',
        'girl': 'person',
        'baby': 'person',
        'toddler': 'person',
        'player': 'person',
        'defender': 'person',
    }
    
    def __init__(self):
        pass
    
    def get_available_objects_for_image(self, image_id: str, qa_space_data: Dict[str, Any], image_dir: Path) -> Tuple[List[str], List[Dict]]:
        """Get available objects for a specific image from actual annotations"""
        # Look for annotation files in the image directory
        annotation_files = []
        
        # Check for annotations subdirectory
        annotations_dir = image_dir / "annotations"
        if annotations_dir.exists():
            # Look for refined annotations first, then regular annotations
            refined_files = list(annotations_dir.glob("*_refined.json"))
            if refined_files:
                annotation_files.extend(refined_files)
            else:
                json_files = list(annotations_dir.glob("*.json"))
                json_files = [f for f in json_files if not f.name.endswith("_refined.json")]
                annotation_files.extend(json_files)
        
        # If no annotations subdirectory, look for JSON files directly in image directory
        if not annotation_files:
            json_files = list(image_dir.glob("*.json"))
            annotation_files.extend(json_files)
        
        if not annotation_files:
            logger.warning(f"No annotation files found for {image_id}")
            return [], []
        
        # Load the first annotation file found
        annotation_file = annotation_files[0]
        try:
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading annotation file {annotation_file}: {e}")
            return [], []
        
        # Extract objects from annotation data
        detected_objects = []
        
        # Handle different annotation formats
        if 'detections' in annotation_data:
            # Refined format
            detections = annotation_data['detections']
        elif 'detected_objects' in annotation_data:
            # Original format
            detections = annotation_data['detected_objects']
        else:
            logger.warning(f"No detections found in annotation file for {image_id}")
            return [], []
        
        # Extract object class names and normalize person-related classes
        normalized_detections = []
        for detection in detections:
            if isinstance(detection, dict):
                class_name = detection.get('class_name', detection.get('class', 'unknown'))
                if class_name and class_name != 'unknown':
                    # Normalize human-related classes to 'person'
                    normalized_class_name = self.PERSON_NORMALIZATION.get(class_name.lower(), class_name)
                    detected_objects.append(normalized_class_name)
                    
                    # Update detection with normalized class name
                    detection_copy = detection.copy()
                    detection_copy['class_name'] = normalized_class_name
                    if 'class' in detection_copy:
                        detection_copy['class'] = normalized_class_name
                    normalized_detections.append(detection_copy)
            elif isinstance(detection, str):
                # Normalize string detections too
                normalized_class_name = self.PERSON_NORMALIZATION.get(detection.lower(), detection)
                detected_objects.append(normalized_class_name)
                normalized_detections.append(normalized_class_name)
        
        # Filter detections by bounding box coverage (remove objects covering >90% of image)
        # Find the original image path to get dimensions
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = image_dir / f"{image_id}{ext}"
            if img_path.exists():
                image_paths.append(img_path)
            # Also check parent directory (openimages_train_10000)
            parent_img_path = Path('/path/to/project/openimages_train_10000') / f"{image_id}{ext}"
            if parent_img_path.exists():
                image_paths.append(parent_img_path)
        
        if image_paths:
            filtered_detections, filtered_out_objects = self.filter_detections_by_coverage(
                normalized_detections, image_paths[0], max_coverage_ratio=0.6
            )
            if filtered_out_objects:
                logger.info(f"Filtered out {len(filtered_out_objects)} objects with oversized bboxes (>60%) in {image_id}: {filtered_out_objects}")
            # Update detected_objects list to match filtered detections
            detected_objects = []
            for detection in filtered_detections:
                if isinstance(detection, dict):
                    class_name = detection.get('class_name', detection.get('class', 'unknown'))
                    if class_name and class_name != 'unknown':
                        detected_objects.append(class_name)
                elif isinstance(detection, str):
                    detected_objects.append(detection)
            normalized_detections = filtered_detections
        
        # Remove duplicates 
        unique_objects = []
        seen = set()
        for obj in detected_objects:
            if obj not in seen:
                unique_objects.append(obj)
                seen.add(obj)
        
        return unique_objects, normalized_detections
    
    def _parse_xyxy_bbox(self, bbox_value: Any) -> Optional[List[float]]:
        """Parse xyxy bbox value which could be list, string representation, or numpy array
        
        Args:
            bbox_value: Bbox value from detection (can be list, string, or numpy array)
            
        Returns:
            List of [x1, y1, x2, y2] floats, or None if invalid
        """
        if bbox_value is None:
            return None
        
        if isinstance(bbox_value, list):
            try:
                return [float(x) for x in bbox_value]
            except (ValueError, TypeError):
                return None
        elif isinstance(bbox_value, str):
            try:
                # Remove brackets and split by whitespace (handles numpy array string format)
                bbox_str = bbox_value.strip('[]')
                values = [float(x) for x in re.split(r'\s+', bbox_str.strip()) if x]
                if len(values) >= 4:
                    return values[:4]
            except (ValueError, TypeError):
                pass
        
        return None
    
    def filter_detections_by_coverage(self, detections: List[Dict], image_path: Path, 
                                     max_coverage_ratio: float = 0.90) -> Tuple[List[Dict], List[str]]:
        """Filter detections based on bounding box coverage of image
        
        Removes detections whose bounding boxes cover more than max_coverage_ratio of the image.
        This is useful for filtering out detections like 'floor' or 'wall' that cover the entire image
        and aren't useful for object-level questions.
        
        Args:
            detections: List of detection dictionaries with bbox information
            image_path: Path to the original image file
            max_coverage_ratio: Maximum allowed coverage ratio (0.0-1.0). Default 0.90 (90%)
            
        Returns:
            Tuple of (filtered_detections, updated_available_objects)
            - filtered_detections: List of detections with coverage <= max_coverage_ratio
            - updated_available_objects: List of unique object class names from filtered detections
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available, skipping bbox coverage filtering")
            # Extract available_objects from unfiltered detections
            available_objects = []
            seen = set()
            for detection in detections:
                if isinstance(detection, dict):
                    class_name = detection.get('class_name', detection.get('class', ''))
                    if class_name and class_name != 'unknown' and class_name not in seen:
                        available_objects.append(class_name)
                        seen.add(class_name)
            return detections, available_objects
        
        # Load image to get dimensions
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}, skipping coverage filtering")
            available_objects = []
            seen = set()
            for detection in detections:
                if isinstance(detection, dict):
                    class_name = detection.get('class_name', detection.get('class', ''))
                    if class_name and class_name != 'unknown' and class_name not in seen:
                        available_objects.append(class_name)
                        seen.add(class_name)
            return detections, available_objects
        
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
                img_area = img_width * img_height
        except Exception as e:
            logger.warning(f"Could not load image {image_path}: {e}, skipping coverage filtering")
            available_objects = []
            seen = set()
            for detection in detections:
                if isinstance(detection, dict):
                    class_name = detection.get('class_name', detection.get('class', ''))
                    if class_name and class_name != 'unknown' and class_name not in seen:
                        available_objects.append(class_name)
                        seen.add(class_name)
            return detections, available_objects
        
        if img_area <= 0:
            logger.warning(f"Invalid image area for {image_path}, skipping coverage filtering")
            available_objects = []
            seen = set()
            for detection in detections:
                if isinstance(detection, dict):
                    class_name = detection.get('class_name', detection.get('class', ''))
                    if class_name and class_name != 'unknown' and class_name not in seen:
                        available_objects.append(class_name)
                        seen.add(class_name)
            return detections, available_objects
        
        # Filter detections based on bbox coverage
        filtered_detections = []
        filtered_count = 0
        
        for detection in detections:
            if not isinstance(detection, dict):
                # Keep non-dict detections (shouldn't happen, but be safe)
                filtered_detections.append(detection)
                continue
            
            # Get bbox coordinates
            bbox = None
            for key in ['xyxy', 'bbox_xyxy', 'bbox', 'bounding_box']:
                if key in detection:
                    bbox = self._parse_xyxy_bbox(detection[key])
                    if bbox:
                        break
            
            if not bbox or len(bbox) < 4:
                # No valid bbox, keep detection (will be handled elsewhere)
                filtered_detections.append(detection)
                continue
            
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Normalize coordinates to image bounds
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            # Calculate coverage ratio
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            coverage_ratio = bbox_area / img_area if img_area > 0 else 0
            
            # Filter based on coverage
            if coverage_ratio <= max_coverage_ratio:
                filtered_detections.append(detection)
            else:
                filtered_count += 1
                class_name = detection.get('class_name', detection.get('class', 'unknown'))
                logger.debug(f"Filtered detection: {class_name} with {coverage_ratio:.2%} coverage (>{max_coverage_ratio:.0%})")
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} detection(s) with >{max_coverage_ratio:.0%} coverage from {image_path.name}")
        
        # Rebuild available_objects from filtered detections
        available_objects = []
        seen = set()
        for detection in filtered_detections:
            if isinstance(detection, dict):
                class_name = detection.get('class_name', detection.get('class', ''))
                if class_name and class_name != 'unknown' and class_name not in seen:
                    available_objects.append(class_name)
                    seen.add(class_name)
        
        return filtered_detections, available_objects
    
    def extract_objects_from_sim_scene(self, scene_path: Path, sm_to_taxonomy: Dict[str, str]) -> Tuple[
        List[str], Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, Any]
    ]:
        """Extract visible objects and their pose data from a sim scene
        
        PRIMARY METHOD: Uses scene_annotations_split.json from processed directory if available.
        FALLBACK: Uses seenable_obj_dict.json + object_annots.json for legacy format.
        
        Returns:
            Tuple of (objects, object_poses, taxonomy_to_sm_names mapping, scene_metadata)
        """
        # Check for new processed format first (primary method)
        processed_annotations_file = scene_path / "scene_annotations_split.json"
        if processed_annotations_file.exists():
            return self._extract_objects_from_processed_scene(scene_path, sm_to_taxonomy)
        
        # Fallback to legacy format
        return self._extract_objects_from_legacy_scene(scene_path, sm_to_taxonomy)
    
    def _extract_objects_from_processed_scene(
        self, scene_path: Path, sm_to_taxonomy: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, Any]]:
        """Extract objects from processed directory format (scene_annotations_split.json)
        
        This is the PRIMARY method for extracting objects from the new processed format.
        The foreground section contains all non-background objects with pre-computed bboxes.
        
        Returns:
            Tuple of (objects, object_poses, taxonomy_to_sm_names mapping, scene_metadata)
        """
        objects = []
        object_poses = {}
        taxonomy_to_sm_names = {}  # taxonomy_name -> list of SM names
        scene_metadata: Dict[str, Any] = {}
        
        annotations_file = scene_path / "scene_annotations_split.json"
        if not annotations_file.exists():
            logger.warning(f"No scene_annotations_split.json found for {scene_path.name}")
            return objects, object_poses, taxonomy_to_sm_names, scene_metadata
        
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations_data = json.load(f)
            logger.debug(f"Loaded scene_annotations_split.json for {scene_path.name}")
        except Exception as e:
            logger.warning(f"Could not load scene_annotations_split.json: {e}")
            return objects, object_poses, taxonomy_to_sm_names, scene_metadata
        
        # Extract foreground objects (non-background objects)
        foreground = annotations_data.get('foreground', {})
        if not foreground:
            logger.warning(f"No foreground objects found in {scene_path.name}")
            return objects, object_poses, taxonomy_to_sm_names, scene_metadata
        
        # Extract camera data for compatibility
        camera_data = annotations_data.get('camera', {})
        scene_metadata = {
            'scene_name': annotations_data.get('scene_name'),
            'view_id': annotations_data.get('view_id'),
            'camera': camera_data,
            'image_path': annotations_data.get('image_path'),
            'annotations_file': str(annotations_file)
        }
        
        # Build seenable_obj_dict equivalent and extract 2D bbox data (skip 3D pose for now)
        seenable_obj_dict = {}
        sm_to_bbox_2d = {}  # sm_name -> bbox_2d
        sm_to_bbox3d = {}
        
        for category, category_objects in foreground.items():
            for obj in category_objects:
                sm_name = obj.get('object_id')
                if not sm_name:
                    continue
                
                # Filter out lighting and debug objects
                if any(x in sm_name.lower() for x in ['light', 'debug', 'capture', 'fog', 'sky', 'game', 'player', 'world', 'landscape', 'exponential', 'atmospheric', 'chaos', 'gameplay']):
                    continue
                
                # Get color (RGB)
                color = obj.get('color')
                if not isinstance(color, list) or len(color) < 3:
                    continue
                
                # Build seenable_obj_dict equivalent
                seenable_obj_dict[sm_name] = color
                
                # Store 2D bbox
                bbox_2d = obj.get('bbox_2d')
                if bbox_2d and len(bbox_2d) >= 4:
                    sm_to_bbox_2d[sm_name] = bbox_2d
                
                # Store 3D bbox information if available
                bbox_3d = obj.get('bbox_3d', {})
                if bbox_3d:
                    sm_to_bbox3d[sm_name] = bbox_3d
        
        logger.debug(f"Extracted {len(seenable_obj_dict)} foreground objects from scene_annotations_split.json")
        
        # Process each foreground object: lookup taxonomy and build object lists
        # Store 2D bbox and available 3D spatial information
        temp_area_key = "_bbox_area"
        for sm_name, color in seenable_obj_dict.items():
            # Direct lookup taxonomy from sm_to_taxonomy (exact match only)
            taxonomy_name = sm_to_taxonomy.get(sm_name)
            if taxonomy_name is None:
                logger.debug(f"Skipping object {sm_name} - no taxonomy mapping found")
                continue
            
            # Get 2D bbox
            bbox_2d = sm_to_bbox_2d.get(sm_name)
            bbox_area = None
            if bbox_2d and len(bbox_2d) >= 4:
                width = max(0.0, float(bbox_2d[2]) - float(bbox_2d[0]))
                height = max(0.0, float(bbox_2d[3]) - float(bbox_2d[1]))
                bbox_area = width * height
            
            # Build pose entry with available spatial data
            pose_entry: Dict[str, Any] = {}
            # Store class_name (taxonomy_name) for high-reliability checks
            pose_entry['class_name'] = taxonomy_name
            if bbox_2d:
                pose_entry['bbox_2d'] = bbox_2d
            if sm_name in sm_to_bbox3d:
                bbox_3d = sm_to_bbox3d[sm_name]
                # Extract accurate 3D location from OBB center (preferred) or AABB center
                # bbox_3d.location is often [0,0,0] or incorrect, so use obb.center or aabb.center instead
                location = None
                if 'obb' in bbox_3d and isinstance(bbox_3d['obb'], dict):
                    obb_center = bbox_3d['obb'].get('center')
                    if obb_center and isinstance(obb_center, list) and len(obb_center) == 3:
                        location = obb_center
                    pose_entry['obb'] = bbox_3d['obb']
                if location is None and 'aabb' in bbox_3d and isinstance(bbox_3d['aabb'], dict):
                    aabb_center = bbox_3d['aabb'].get('center')
                    if aabb_center and isinstance(aabb_center, list) and len(aabb_center) == 3:
                        location = aabb_center
                    pose_entry['aabb'] = bbox_3d['aabb']
                # Fallback to bbox_3d.location only if obb/aabb centers not available
                if location is None:
                    location = bbox_3d.get('location')
                
                # Store accurate location
                if location and isinstance(location, list) and len(location) == 3:
                    # Validate that location is not [0, 0, 0] (common placeholder)
                    if location != [0, 0, 0]:
                        pose_entry['location'] = location
                    else:
                        logger.debug(f"Warning: {sm_name} has location [0,0,0], skipping (no valid 3D center found)")
                
                # Copy rotation and scale if present
                rotation = bbox_3d.get('rotation')
                scale = bbox_3d.get('scale')
                if rotation and isinstance(rotation, list):
                    pose_entry['rotation'] = rotation
                    # Convert rotation to orientation vectors (left/front) for geometric spatial reasoning
                    try:
                        from modules.qa_modules.bbox3d_utils import rot_mat
                        import numpy as np
                        rotation_matrix = rot_mat(rotation)
                        # Extract orientation vectors from rotation matrix
                        # Sim image coordinate system: X=left/right, Y=front/behind, Z=above/below
                        # Rotation matrix columns: [0]=world X (left/right), [1]=world Y (up/down), [2]=world Z (forward/backward)
                        # Mapping: Column 0 (world X) → sim X (left/right) = left vector
                        #         Column 2 (world Z/forward) → sim Y (front/behind) = front vector
                        left_vec = rotation_matrix[:, 0].tolist()
                        front_vec = rotation_matrix[:, 2].tolist()
                        pose_entry['left'] = left_vec
                        pose_entry['front'] = front_vec
                    except Exception as e:
                        logger.debug(f"Could not convert rotation to vectors for {sm_name}: {e}")
                if scale and isinstance(scale, list):
                    pose_entry['scale'] = scale
            pose_entry[temp_area_key] = bbox_area if bbox_area is not None else -1.0
            pose_entry['sm_name'] = sm_name
            
            # Group by taxonomy name (handle multiple SM instances -> same taxonomy)
            if taxonomy_name not in taxonomy_to_sm_names:
                taxonomy_to_sm_names[taxonomy_name] = [sm_name]
                objects.append(taxonomy_name)
                if pose_entry:
                    object_poses[taxonomy_name] = pose_entry
            else:
                # Multiple SM instances for same taxonomy - append to list
                if sm_name not in taxonomy_to_sm_names[taxonomy_name]:
                    taxonomy_to_sm_names[taxonomy_name].append(sm_name)
                    logger.debug(f"Added duplicate instance {sm_name} for {taxonomy_name} (total: {len(taxonomy_to_sm_names[taxonomy_name])})")
                if pose_entry:
                    existing_pose = object_poses.get(taxonomy_name)
                    if existing_pose is None or pose_entry[temp_area_key] > existing_pose.get(temp_area_key, -1.0):
                        object_poses[taxonomy_name] = pose_entry
        
        # Clean temporary fields used for comparison
        for pose in object_poses.values():
            pose.pop(temp_area_key, None)
        
        logger.info(f"Found {len(objects)} objects from processed scene for {scene_path.name}: {objects[:5]}...")
        return objects, object_poses, taxonomy_to_sm_names, scene_metadata
    
    def _extract_objects_from_legacy_scene(
        self, scene_path: Path, sm_to_taxonomy: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, Any]]:
        """Extract objects from legacy format (seenable_obj_dict.json + object_annots.json)
        
        This is the FALLBACK method for legacy scene directories.
        
        Returns:
            Tuple of (objects, object_poses, taxonomy_to_sm_names mapping, scene_metadata)
        """
        objects = []
        object_poses = {}
        taxonomy_to_sm_names = {}  # taxonomy_name -> list of SM names
        scene_metadata: Dict[str, Any] = {
            'image_path': str((scene_path / "lit.png").resolve()) if (scene_path / "lit.png").exists() else None,
            'annotations_file': str((scene_path / "object_annots.json").resolve()) if (scene_path / "object_annots.json").exists() else None
        }
        
        # Load seenable_obj_dict.json - this is the source of truth for visible objects
        seenable_obj_file = scene_path / "seenable_obj_dict.json"
        if not seenable_obj_file.exists():
            logger.warning(f"No seenable_obj_dict.json found for {scene_path.name}")
            return objects, object_poses, taxonomy_to_sm_names, scene_metadata
        
        try:
            with open(seenable_obj_file, 'r') as f:
                seenable_obj_dict = json.load(f)
            logger.debug(f"Loaded seenable_obj_dict with {len(seenable_obj_dict)} objects")
        except Exception as e:
            logger.warning(f"Could not load seenable_obj_dict: {e}")
            return objects, object_poses, taxonomy_to_sm_names, scene_metadata
        
        # Load camera annotations for 3D projection (optional)
        camera_annots_file = scene_path / "camera_annots.json"
        camera_annots_data = {}
        if camera_annots_file.exists():
            try:
                with open(camera_annots_file, 'r', encoding='utf-8') as f:
                    camera_annots_data = json.load(f)
                scene_metadata['camera'] = camera_annots_data
            except Exception as e:
                logger.warning(f"Could not load camera annotations: {e}")
        
        # Load object annotations with pose data (to match SM names)
        object_annots_file = scene_path / "object_annots.json"
        object_annots_map = {}  # obj_id -> obj_data
        if object_annots_file.exists():
            try:
                with open(object_annots_file, 'r', encoding='utf-8') as f:
                    annots_data = json.load(f)
                # Build lookup map: SM_name -> obj_data
                for obj_data in annots_data.get('outputs', []):
                    obj_id = obj_data.get('object_id', '')
                    if obj_id:
                        object_annots_map[obj_id] = obj_data
                logger.debug(f"Loaded {len(object_annots_map)} objects from object_annots.json")
            except Exception as e:
                logger.warning(f"Could not load object annotations: {e}")
        
        # Import 3D bbox utilities
        from modules.qa_modules.bbox3d_utils import compute_2d_bbox_from_3d_pose
        
        # Start from seenable_obj_dict - iterate through visible SM names
        for sm_name, color in seenable_obj_dict.items():
            # Skip invalid entries
            if not isinstance(color, list) or len(color) < 3:
                continue
            
            # Filter out lighting and debug objects
            if any(x in sm_name.lower() for x in ['light', 'debug', 'capture', 'fog', 'sky', 'game', 'player', 'world', 'landscape', 'exponential', 'atmospheric', 'chaos', 'gameplay']):
                continue
            
            # Direct lookup taxonomy from sm_to_taxonomy (exact match only)
            taxonomy_name = sm_to_taxonomy.get(sm_name)
            if taxonomy_name is None:
                # No mapping found - skip this object
                logger.debug(f"Skipping object {sm_name} - no taxonomy mapping found")
                continue
            
            # Get pose data from object_annots (exact SM name match)
            obj_data = object_annots_map.get(sm_name)
            pose_data = None
            if obj_data:
                # Check if object has meaningful bounds (not zero extent)
                bounds = obj_data.get('bounds', {})
                extent = bounds.get('extent', [0, 0, 0])
                if any(x > 0 for x in extent):
                    pose_data = {
                        'location': obj_data.get('location', [0, 0, 0]),
                        'rotation': obj_data.get('rotation', [0, 0, 0]),
                        'scale': obj_data.get('scale', [1, 1, 1]),
                        'bounds': bounds,
                        'aabb': obj_data.get('aabb', {}),
                        'obb': obj_data.get('obb', {}),
                        'class_name': taxonomy_name
                    }
                    
                    # Convert rotation to orientation vectors (left/front) for geometric spatial reasoning
                    rotation = pose_data.get('rotation')
                    if rotation and isinstance(rotation, list):
                        try:
                            from modules.qa_modules.bbox3d_utils import rot_mat
                            import numpy as np
                            rotation_matrix = rot_mat(rotation)
                            # Extract orientation vectors from rotation matrix
                            # Sim image coordinate system: X=left/right, Y=front/behind, Z=above/below
                            # Rotation matrix columns: [0]=world X (left/right), [1]=world Y (up/down), [2]=world Z (forward/backward)
                            # Mapping: Column 0 (world X) → sim X (left/right) = left vector
                            #         Column 2 (world Z/forward) → sim Y (front/behind) = front vector
                            left_vec = rotation_matrix[:, 0].tolist()
                            front_vec = rotation_matrix[:, 2].tolist()
                            pose_data['left'] = left_vec
                            pose_data['front'] = front_vec
                        except Exception as e:
                            logger.debug(f"Could not convert rotation to vectors for {sm_name}: {e}")
                    
                    # Compute 2D bbox from 3D pose using camera parameters
                    if camera_annots_data:
                        try:
                            bbox_2d, visible = compute_2d_bbox_from_3d_pose(pose_data, camera_annots_data)
                            if bbox_2d and visible:
                                pose_data['bbox_2d'] = bbox_2d
                        except Exception as e:
                            logger.debug(f"Could not compute 3D bbox for {sm_name}: {e}")
            else:
                # No pose data found - object still visible but no pose
                logger.debug(f"No pose data found for {sm_name}, will skip for spatial reasoning")
            
            # Group by taxonomy name (handle multiple SM instances -> same taxonomy)
            if taxonomy_name not in taxonomy_to_sm_names:
                taxonomy_to_sm_names[taxonomy_name] = []
                objects.append(taxonomy_name)
                # Use first SM instance's pose (if available)
                if pose_data:
                    object_poses[taxonomy_name] = pose_data
            else:
                # Multiple SM instances for same taxonomy - append to list
                if sm_name not in taxonomy_to_sm_names[taxonomy_name]:
                    taxonomy_to_sm_names[taxonomy_name].append(sm_name)
                    logger.debug(f"Added duplicate instance {sm_name} for {taxonomy_name} (total: {len(taxonomy_to_sm_names[taxonomy_name])})")
        
        logger.info(f"Found {len(objects)} objects from legacy scene for {scene_path.name}: {objects[:5]}...")
        return objects, object_poses, taxonomy_to_sm_names, scene_metadata
