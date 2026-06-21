import os
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VisualizationUtils:
    @staticmethod
    def resize_and_save_image(image_path: str, output_path: Path, target_width: int = 400, target_height: int = 225) -> str:
        """Resize and save image to consistent dimensions"""
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return ""
        
        try:
            image = Image.open(image_path).convert('RGB')
            original_width, original_height = image.size
            
            # Calculate scaling factor to fit within target dimensions while maintaining aspect ratio
            scale_w = target_width / original_width
            scale_h = target_height / original_height
            scale = min(scale_w, scale_h)  # Use smaller scale to ensure image fits within target dimensions
            
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize the image
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height} (scale: {scale:.3f})")
            
            # Save the image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=100)
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error resizing image {image_path}: {e}")
            return ""

    @staticmethod
    def draw_2d_bbox_image(image_path: str, detected_objects: List[Dict], output_path: Path, target_width: int = 400, target_height: int = 225) -> str:
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return ""
        
        image = Image.open(image_path).convert('RGB')
        
        # Store original size for scaling
        original_width, original_height = image.size
        img_width, img_height = original_width, original_height
        
        # Draw bboxes on ORIGINAL size image first, then resize the whole thing
        draw = ImageDraw.Draw(image)
        
        # Colors prioritized for VLM recognition: most distinct, obvious, unambiguous names
        # Primary colors (RED, GREEN, BLUE, YELLOW, ORANGE, PINK, PURPLE) are clearest for VLMs
        # RGB values designed to be visually distinct and easily recognizable
        colors = [
            (255, 0, 0),      # Red - PRIORITY: most obvious, distinct
            (0, 255, 0),      # Green - PRIORITY: most obvious, distinct
            (0, 0, 255),      # Blue - PRIORITY: most obvious, distinct
            (255, 255, 0),    # Yellow - PRIORITY: most obvious, distinct
            (255, 165, 0),    # Orange - PRIORITY: most obvious, distinct
            (255, 192, 203),  # Pink - PRIORITY: most obvious, distinct
            (128, 0, 128),    # Purple - PRIORITY: most obvious, distinct
            # Fallback colors (used only if more than 7 objects)
            (255, 0, 255),    # Magenta - fallback
            (0, 255, 255),    # Cyan - fallback
            (255, 20, 147),   # Deep Pink - fallback
            (138, 43, 226),   # Blue Violet - fallback
            (64, 224, 208),   # Turquoise - fallback
        ]
        
        # Color names prioritized: PRIMARY colors are clearest for VLM text recognition
        # VLMs see only the text name, not RGB values
        # Primary colors (red, green, blue, yellow, orange, pink, purple) are most unambiguous
        color_names = [
            "red",           # PRIORITY: Standard, obvious, distinct
            "green",         # PRIORITY: Standard, obvious, distinct
            "blue",          # PRIORITY: Standard, obvious, distinct
            "yellow",        # PRIORITY: Standard, obvious, distinct
            "orange",        # PRIORITY: Standard, obvious, distinct
            "pink",          # PRIORITY: Standard, obvious, distinct
            "purple",        # PRIORITY: Standard, obvious, distinct
            # Fallback colors (used only if more than 7 objects)
            "magenta",       # Fallback
            "cyan",          # Fallback
            "rose",          # Fallback
            "violet",        # Fallback
            "turquoise",     # Fallback
        ]
        
        # Bboxes are already in original image coordinates - no scaling needed yet
        scale_x = 1.0
        scale_y = 1.0
        
        # First pass: collect all valid bboxes and calculate their scaled coordinates
        # NOTE: The order of detected_objects determines color assignment
        # Colors are assigned by index: 0=red, 1=green, 2=blue, 3=yellow, 4=magenta, 5=cyan, etc.
        # This order MUST match the order used in questions (available_objects/choices)
        valid_objects = []
        for idx, obj in enumerate(detected_objects):
            bbox = obj.get('bbox', obj.get('bounding_box', obj.get('xyxy', [])))
            logger.debug(f"Processing object {idx}: bbox={bbox}")
            
            if not bbox or len(bbox) < 4:
                logger.warning(f"Object {idx}: invalid bbox {bbox}")
                continue
            
            if isinstance(bbox, str):
                try:
                    bbox_str = bbox.strip('[]')
                    bbox = [float(x.strip()) for x in bbox_str.split() if x.strip()]
                except:
                    continue
            
            if len(bbox) < 4:
                continue
                
            x1, y1, x2, y2 = bbox[:4]
            
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            
            logger.debug(f"Object {idx}: scaled from [{x1}, {y1}, {x2}, {y2}] to [{x1_scaled}, {y1_scaled}, {x2_scaled}, {y2_scaled}]")
            
            # Color assignment by index - MUST match question color order
            # RGB order: (255,0,0)=red, (0,255,0)=green, (0,0,255)=blue, (255,255,0)=yellow, (255,0,255)=magenta, (0,255,255)=cyan
            color = colors[idx % len(colors)]
            color_name = color_names[idx % len(color_names)]
            obj['object_number'] = str(idx + 1)
            # Save color info directly in the detection object
            obj['_bbox_color_rgb'] = color
            obj['_bbox_color_name'] = color_name
            obj['_bbox_color_index'] = idx
            
            valid_objects.append({
                'idx': idx,
                'obj': obj,
                'bbox_scaled': (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
                'color': color,
                'color_name': color_name
            })
        
        # Second pass: draw all rectangles first
        for obj_data in valid_objects:
            x1_scaled, y1_scaled, x2_scaled, y2_scaled = obj_data['bbox_scaled']
            color = obj_data['color']
            idx = obj_data['idx']
            
            logger.debug(f"Drawing rectangle for object {idx} with color {color}")
            draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], outline=color, width=4)
        
        # Now resize the image with bboxes already drawn to target size
        # This ensures bboxes scale correctly with the image
        max_size = max(target_width, target_height)
        # Force resize if image is larger than target
        if max(original_width, original_height) > max_size:
            # Calculate new dimensions maintaining aspect ratio
            if original_width > original_height:
                new_width = max_size
                new_height = int((original_height * max_size) / original_width)
            else:
                new_height = max_size
                new_width = int((original_width * max_size) / original_height)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Resized bbox image from {original_width}x{original_height} to {new_width}x{new_height}")
        else:
            logger.info(f"Image is already within target size: {original_width}x{original_height}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)
        logger.info(f"  Saved 2D bbox image with {len(detected_objects)} objects")
        
        return str(output_path)
    
    @staticmethod
    def extract_bboxes_from_segmentation(seg_path: str, object_annots: Dict[str, Any] = None) -> List[Dict]:
        """Extract bounding boxes from segmentation image for sim images"""
        try:
            seg_img = Image.open(seg_path)
            seg_array = np.array(seg_img)
            
            # Convert to grayscale if needed
            if len(seg_array.shape) == 3:
                if seg_array.shape[2] == 4:  # RGBA
                    # Convert to uint32 first to avoid overflow
                    seg_rgb = seg_array[:, :, :3].astype(np.uint32)
                    seg_gray = (seg_rgb[:, :, 0] * 256 * 256 + 
                               seg_rgb[:, :, 1] * 256 + 
                               seg_rgb[:, :, 2]).astype(np.uint32)
                else:
                    seg_gray = seg_array[:, :, 0].astype(np.uint32)
            else:
                seg_gray = seg_array.astype(np.uint32)
            
            unique_segments = np.unique(seg_gray)
            detected_objects = []
            
            # For each segment, find bounding box
            for segment_id in unique_segments:
                if segment_id == 0:  # Background
                    continue
                
                # Find all pixels belonging to this segment
                mask = (seg_gray == segment_id)
                if not np.any(mask):
                    continue
                
                # Prefer connected-component bounding boxes to avoid merging disjoint islands
                try:
                    from scipy import ndimage as ndi
                    labeled, num = ndi.label(mask)
                    if num == 0:
                        continue
                    component_ids = list(range(1, num + 1))
                    for cid in component_ids:
                        comp_mask = (labeled == cid)
                        rows = np.any(comp_mask, axis=1)
                        cols = np.any(comp_mask, axis=0)
                        if not (np.any(rows) and np.any(cols)):
                            continue
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min
                        bbox_area = bbox_width * bbox_height
                        # Filter small
                        min_bbox_width = 40
                        min_bbox_height = 40
                        if bbox_width < min_bbox_width or bbox_height < min_bbox_height:
                            continue
                        detected_objects.append({
                            'segment_id': int(segment_id),
                            'bbox': [x_min, y_min, x_max, y_max],
                            'bbox_area': bbox_area,
                            'class_name': f'object_{segment_id}'
                        })
                except Exception:
                    # Fallback: single bbox for the whole segment
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if np.any(rows) and np.any(cols):
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min
                        bbox_area = bbox_width * bbox_height
                        min_bbox_width = 40
                        min_bbox_height = 40
                        if bbox_width < min_bbox_width or bbox_height < min_bbox_height:
                            continue
                        detected_objects.append({
                            'segment_id': int(segment_id),
                            'bbox': [x_min, y_min, x_max, y_max],
                            'bbox_area': bbox_area,
                            'class_name': f'object_{segment_id}'
                        })
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error extracting bboxes from segmentation: {e}")
            return []
    
    @staticmethod
    def create_bbox_image_from_segmentation(lit_path: str, seg_path: str, output_path: Path, 
                                            target_width: int = 300, target_height: int = 400):
        """Create bounding box visualization from sim image segmentation"""
        try:
            # Load original image - DON'T resize yet, draw on original size first
            lit_img = Image.open(lit_path).convert('RGB')
            original_width, original_height = lit_img.size
            
            # Calculate scaling factor
            scale_w = target_width / original_width
            scale_h = target_height / original_height
            scale = min(scale_w, scale_h)
            
            # Extract bounding boxes from segmentation (at original size)
            bboxes = VisualizationUtils.extract_bboxes_from_segmentation(seg_path)
            
            # Draw bounding boxes on ORIGINAL size image first
            if bboxes:
                draw = ImageDraw.Draw(lit_img)
                colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0)
                ]
                
                # First pass: collect all bboxes with ORIGINAL coordinates (no scaling yet)
                valid_bboxes = []
                for idx, bbox_data in enumerate(bboxes):
                    x_min, y_min, x_max, y_max = bbox_data['bbox']
                    
                    # Use original coordinates - no scaling
                    x_min_orig = int(x_min)
                    y_min_orig = int(y_min)
                    x_max_orig = int(x_max)
                    y_max_orig = int(y_max)
                    
                    color = colors[idx % len(colors)]
                    valid_bboxes.append({
                        'bbox_orig': (x_min_orig, y_min_orig, x_max_orig, y_max_orig),
                        'color': color,
                        'idx': idx
                    })
                
                # Second pass: draw all rectangles first
                for bbox_info in valid_bboxes:
                    x_min, y_min, x_max, y_max = bbox_info['bbox_orig']
                    color = bbox_info['color']
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=4)
                
                # Resize the image with bboxes drawn (labels will be drawn after resize)
                max_size = max(target_width, target_height)
                scale_x, scale_y = 1.0, 1.0
                if max(original_width, original_height) > max_size:
                    # Calculate new dimensions maintaining aspect ratio
                    if original_width > original_height:
                        new_width = max_size
                        new_height = int((original_height * max_size) / original_width)
                        scale_x = scale_y = max_size / original_width  # Same scale for both (width-constrained)
                    else:
                        new_height = max_size
                        new_width = int((original_width * max_size) / original_height)
                        scale_x = scale_y = max_size / original_height  # Same scale for both (height-constrained)
                    
                    lit_img = lit_img.resize((new_width, new_height), Image.LANCZOS)
                    logger.info(f"Resized bbox image from {original_width}x{original_height} to {new_width}x{new_height}")
                else:
                    logger.info(f"Image is already within target size: {original_width}x{original_height}")
                    new_width, new_height = original_width, original_height
                
                # Save the image only if bboxes exist
                output_path.parent.mkdir(parents=True, exist_ok=True)
                lit_img.save(output_path, quality=95)
                logger.info(f"Created bbox image from segmentation with {len(bboxes)} objects")
            else:
                logger.warning(f"No bounding boxes found for {seg_path}, skipping bbox image creation")
                return
            
        except Exception as e:
            logger.error(f"Error creating bbox image from segmentation: {e}")
            return

