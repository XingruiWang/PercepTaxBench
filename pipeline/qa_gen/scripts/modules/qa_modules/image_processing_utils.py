#!/usr/bin/env python3

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)


class ImageProcessingUtils:
    """Utilities for processing images and creating visualizations for QA benchmarks"""
    
    @staticmethod
    def copy_images_with_bbox(image_id: str, source_dir: Path, target_images_dir: Path, 
                             detections: List[Dict], original_images_dir: Path = None) -> None:
        """Copy original image and create bounding box visualization"""
        try:
            # Create image subdirectory
            image_output_dir = target_images_dir / image_id
            image_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find original image file
            if original_images_dir:
                # Use original images directory (for real images)
                original_image_path = original_images_dir / f"{image_id}.jpg"
                if original_image_path.exists():
                    image_files = [original_image_path]
                else:
                    # Fallback: look for original images in source directory
                    image_files = ImageProcessingUtils._find_original_images_in_source(source_dir)
            else:
                # Use source directory directly (for sim images)
                image_files = ImageProcessingUtils._find_original_images_in_source(source_dir)
            
            if not image_files:
                logger.warning(f"No image file found for {image_id}")
                return
            
            original_image_path = image_files[0]  # Take first image file found
            
            # Copy and resize original image using VisualizationUtils
            target_original = image_output_dir / "original.jpg"
            VisualizationUtils.resize_and_save_image(str(original_image_path), target_original)
            
            # Create bounding box visualization
            target_bbox = image_output_dir / "bbox.jpg"
            
            # For sim images, check if we have segmentation data
            if not original_images_dir:  # Sim images don't have original_images_dir
                seg_path = source_dir / "seg.png"
                if seg_path.exists():
                    # Use segmentation-based bounding boxes
                    VisualizationUtils.create_bbox_image_from_segmentation(
                        str(original_image_path), str(seg_path), target_bbox
                    )
                else:
                    # Fallback: no segmentation, just resize
                    VisualizationUtils.resize_and_save_image(str(original_image_path), target_bbox)
            else:
                # Real images: use detections
                if detections:
                    VisualizationUtils.draw_2d_bbox_image(str(original_image_path), detections, target_bbox)
                else:
                    VisualizationUtils.resize_and_save_image(str(original_image_path), target_bbox)
            
        except Exception as e:
            logger.error(f"Error copying images for {image_id}: {e}")
    
    @staticmethod
    def _find_original_images_in_source(source_dir: Path) -> List[Path]:
        """Find original images in source directory (excluding visualizations)"""
        original_files = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            files = list(source_dir.glob(ext))
            # Filter out visualization files
            original_files.extend([f for f in files if "visualization" not in f.name.lower() and "3d" not in f.name.lower()])
        
        if original_files:
            return original_files
        else:
            # Last resort: look in visualizations subdirectory
            viz_dir = source_dir / "visualizations"
            if viz_dir.exists():
                viz_files = list(viz_dir.glob("*.jpg")) + list(viz_dir.glob("*.png"))
                # Prefer non-3D visualization files if available
                non_3d_viz = [f for f in viz_files if "3d" not in f.name.lower()]
                if non_3d_viz:
                    return non_3d_viz
                else:
                    return viz_files
        
        return []
    
    @staticmethod
    def copy_images_with_bbox_group(scene_id: str, source_dir: Path, target_images_dir: Path, 
                                    group_objects: List[str], group_idx: int, 
                                    object_poses: Dict[str, Dict[str, Any]] = None,
                                    taxonomy_to_sm_names: Dict[str, List[str]] = None,
                                    validated_bboxes: Dict[str, Dict] = None,
                                    scene_metadata: Dict[str, Any] = None) -> List[str]:
        """Copy original image and create bounding box visualization for a specific group of objects.
        
        Uses segmentation-based bbox extraction (most reliable). Falls back to resized image if no seg data.
        
        Args:
            taxonomy_to_sm_names: Mapping from taxonomy object names to list of SM names in the scene
        """
        try:
            # Create group subdirectory
            group_output_dir = target_images_dir / f"{scene_id}_group{group_idx}"
            group_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine source image path (clean image without bboxes)
            image_override = None
            if scene_metadata:
                image_meta_path = scene_metadata.get('image_path')
                if image_meta_path:
                    image_override = Path(image_meta_path)
                    if not image_override.exists():
                        logger.warning(f"Scene metadata image_path does not exist: {image_override}")
                        image_override = None
            
            # MUST use lit.png or override path - do not use visualization images with pre-drawn bboxes
            lit_image_path = image_override if image_override else source_dir / "lit.png"
            seg_image_path = source_dir / "seg.png"
            
            # Check for processed format (scene_annotations_split.json)
            processed_format = (source_dir / "scene_annotations_split.json").exists()
            if processed_format:
                # Use override if provided; otherwise expect lit.png in source_dir
                if not lit_image_path.exists():
                    logger.warning(
                        f"No base image found for {scene_id} (expected {lit_image_path}) - skipping image to avoid using pre-drawn bbox images"
                    )
                    return []  # Return empty list instead of None
                else:
                    logger.debug(f"Using base image for {scene_id}: {lit_image_path}")
            
            if not lit_image_path.exists():
                logger.warning(f"No lit.png found for {scene_id}")
                return []  # Return empty list instead of None
            
            # Copy and resize original image
            target_original = group_output_dir / "original.jpg"
            VisualizationUtils.resize_and_save_image(str(lit_image_path), target_original)
            
            # Create bbox visualization with only the objects in this group
            target_bbox = group_output_dir / "bbox.jpg"
            
            # For processed format, use pre-computed bbox_2d from scene_annotations_split.json
            if processed_format:
                # Load scene_annotations_split.json to get bbox_2d data
                annotations_file = source_dir / "scene_annotations_split.json"
                if annotations_file.exists():
                    try:
                        with open(annotations_file, 'r') as f:
                            annotations_data = json.load(f)
                        
                        # Extract foreground objects and build detection list
                        foreground = annotations_data.get('foreground', {})
                        detected_objects = []
                        
                        # Load SM to taxonomy mapping
                        from modules.qa_modules.data_loading_utils import DataLoadingUtils
                        data_loader = DataLoadingUtils()
                        sm_to_taxonomy = data_loader.load_sm_to_human_mapping()
                        
                        # Build group SM names set
                        group_sm_names = set()
                        for obj_name in group_objects:
                            if obj_name in taxonomy_to_sm_names:
                                group_sm_names.update(taxonomy_to_sm_names[obj_name])
                        
                        # Extract bbox_2d from foreground objects
                        # Only include objects that are in group_objects (filter by taxonomy name)
                        group_taxonomy_names = set(group_objects)  # Fast lookup
                        taxonomy_to_best_bbox = {}  # taxonomy_name -> best bbox (largest area)
                        
                        # NO SCALING: Use original image size and original bbox coordinates
                        from PIL import Image
                        draw_img = Image.open(lit_image_path)
                        img_width, img_height = draw_img.size
                        
                        logger.info(f"Using original image size: {img_width}x{img_height}, drawing bboxes with original coordinates (no scaling)")
                        
                        for category, category_objects in foreground.items():
                            for obj in category_objects:
                                sm_name = obj.get('object_id')
                                if not sm_name:
                                    continue
                                
                                # Map SM name to taxonomy name
                                taxonomy_name = sm_to_taxonomy.get(sm_name, sm_name)
                                
                                # Only process if this taxonomy object is in the group
                                if taxonomy_name not in group_taxonomy_names:
                                    continue
                                
                                # Verify this SM instance is actually in group_sm_names (safety check)
                                if sm_name not in group_sm_names:
                                    continue
                                
                                bbox_2d = obj.get('bbox_2d')
                                if bbox_2d and len(bbox_2d) >= 4:
                                    # SIM IMAGE: Use original bbox_2d coordinates directly (NO SCALING)
                                    x1, y1, x2, y2 = bbox_2d[:4]
                                    
                                    # Convert to integers (no scaling)
                                    x1_draw = int(x1)
                                    y1_draw = int(y1)
                                    x2_draw = int(x2)
                                    y2_draw = int(y2)
                                    
                                    # Clamp coordinates to valid image bounds
                                    x1_draw = max(0, min(x1_draw, img_width - 1))
                                    y1_draw = max(0, min(y1_draw, img_height - 1))
                                    x2_draw = max(0, min(x2_draw, img_width - 1))
                                    y2_draw = max(0, min(y2_draw, img_height - 1))
                                    
                                    # Ensure valid bbox (x1 < x2, y1 < y2)
                                    if x1_draw >= x2_draw or y1_draw >= y2_draw:
                                        logger.debug(f"Skipping invalid bbox for {taxonomy_name}: [{x1_draw}, {y1_draw}, {x2_draw}, {y2_draw}]")
                                        continue
                                    
                                    bbox_final = [x1_draw, y1_draw, x2_draw, y2_draw]
                                    
                                    # Calculate bbox area (using original coordinates)
                                    area = (x2 - x1) * (y2 - y1)
                                    
                                    # Keep best bbox (largest area) for each taxonomy object
                                    if taxonomy_name not in taxonomy_to_best_bbox:
                                        taxonomy_to_best_bbox[taxonomy_name] = {
                                            'bbox': bbox_final,
                                            'class_name': taxonomy_name,
                                            'area': area
                                        }
                                    else:
                                        # Update if this bbox is larger
                                        if area > taxonomy_to_best_bbox[taxonomy_name]['area']:
                                            taxonomy_to_best_bbox[taxonomy_name] = {
                                                'bbox': bbox_final,
                                                'class_name': taxonomy_name,
                                                'area': area
                                            }
                        
                        # Convert to list, maintaining order from group_objects
                        # CRITICAL: Only include objects that are in group_objects (filter strictly)
                        detected_objects = []
                        group_objects_set = set(group_objects)  # Fast lookup
                        
                        for obj_name in group_objects:
                            # Only add if this object is in the group AND has a bbox
                            if obj_name in group_objects_set and obj_name in taxonomy_to_best_bbox:
                                detected_objects.append({
                                    'bbox': taxonomy_to_best_bbox[obj_name]['bbox'],
                                    'class_name': obj_name
                                })
                        
                        # Double-check: ensure we only have objects from group_objects
                        detected_object_names = {obj['class_name'] for obj in detected_objects}
                        if detected_object_names - group_objects_set:
                            logger.warning(f"Warning: detected_objects contains objects not in group: {detected_object_names - group_objects_set}")
                            # Filter out any objects not in group_objects
                            detected_objects = [obj for obj in detected_objects if obj['class_name'] in group_objects_set]
                        
                        if detected_objects:
                            detected_objects = ImageProcessingUtils._filter_overlapping_detections(
                                detected_objects,
                                iou_threshold=0.2,
                                coverage_threshold=0.85,
                                scene_id=scene_id,
                                group_idx=group_idx
                            )
                        
                        if detected_objects:
                            # SIM IMAGE: Draw bboxes on original image using original coordinates (NO SCALING)
                            logger.debug(f"Drawing bboxes for group {group_idx}: {[obj['class_name'] for obj in detected_objects]}")
                            
                            from PIL import ImageDraw
                            draw = ImageDraw.Draw(draw_img)
                            colors = [
                                (255, 0, 0),      # Red
                                (0, 255, 0),      # Green
                                (0, 0, 255),      # Blue
                                (255, 255, 0),    # Yellow
                                (255, 165, 0),    # Orange
                                (255, 192, 203),  # Pink
                                (128, 0, 128),    # Purple
                            ]
                            color_names = [
                                "red", "green", "blue", "yellow", "orange", "pink", "purple",
                                "magenta", "cyan", "rose", "violet", "turquoise"
                            ]
                            
                            for idx, obj in enumerate(detected_objects):
                                bbox = obj.get('bbox', [])
                                if len(bbox) >= 4:
                                    x1, y1, x2, y2 = bbox[:4]
                                    color = colors[idx % len(colors)]
                                    logger.debug(f"Drawing bbox for {obj.get('class_name')}: [{x1}, {y1}, {x2}, {y2}] on {img_width}x{img_height} image")
                                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                            
                            # Save at original image size (NO RESIZE)
                            target_bbox.parent.mkdir(parents=True, exist_ok=True)
                            # Ensure RGB mode (JPEG doesn't support RGBA)
                            if draw_img.mode != 'RGB':
                                draw_img = draw_img.convert('RGB')
                            
                            # Resize bbox image to match original.jpg size (400x225)
                            target_width = 400
                            target_height = 225
                            original_width, original_height = draw_img.size
                            
                            # Calculate scaling factor to fit within target dimensions while maintaining aspect ratio
                            scale_w = target_width / original_width
                            scale_h = target_height / original_height
                            scale = min(scale_w, scale_h)
                            
                            new_width = int(original_width * scale)
                            new_height = int(original_height * scale)
                            
                            # Resize the image with bboxes (bboxes will scale automatically)
                            draw_img = draw_img.resize((new_width, new_height), Image.LANCZOS)
                            logger.info(f"SIM IMAGE: Resized bbox image from {original_width}x{original_height} to {new_width}x{new_height} (scale: {scale:.3f})")
                            draw_img.save(target_bbox, quality=95)
                            
                            # Save color mapping for question generation
                            try:
                                sidecar = target_bbox.parent / "objects_used.json"
                                objects_with_boxes = {}
                                for idx, obj in enumerate(detected_objects):
                                    color_name = color_names[idx % len(color_names)]
                                    # Capitalize first letter of each word (matches real image format)
                                    color_name_cap = ' '.join(word.capitalize() for word in color_name.split())
                                    box_label = f"{color_name_cap} box"
                                    objects_with_boxes[box_label] = obj['class_name']
                                
                                output_data = {
                                    "box_mapping": objects_with_boxes,
                                    "objects": [obj['class_name'] for obj in detected_objects]
                                }
                                with open(sidecar, 'w') as f:
                                    json.dump(output_data, f, indent=2)
                            except Exception as e:
                                logger.warning(f"Failed to save color mapping: {e}")
                            
                            # Return list of objects that have bboxes (only from this group)
                            objects_with_bboxes = [obj['class_name'] for obj in detected_objects]
                            logger.info(f"Created bbox image for {scene_id}_group{group_idx} with {len(objects_with_bboxes)} objects using processed format")
                            
                            return objects_with_bboxes
                        else:
                            # Fallback: copy visualization image
                            import shutil
                            shutil.copy2(lit_image_path, target_bbox)
                            logger.info(f"No bboxes found for group, copied visualization image")
                            return []  # Return empty list if no bboxes found
                    except Exception as e:
                        logger.warning(f"Error loading processed format annotations: {e}")
                        # Fallback: copy visualization image
                        import shutil
                        shutil.copy2(lit_image_path, target_bbox)
                        return []  # Return empty list on error
            elif seg_image_path.exists():
                logger.debug(f"Extracting bboxes from segmentation for {len(group_objects)} group objects")
                
                # Load seenable_obj_dict.json (SM_name -> color mapping)
                seenable_obj_file = source_dir / "seenable_obj_dict.json"
                seenable_obj_data = {}
                if seenable_obj_file.exists():
                    try:
                        with open(seenable_obj_file, 'r') as f:
                            seenable_obj_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load seenable_obj_dict: {e}")
                
                if seenable_obj_data and taxonomy_to_sm_names:
                    # Prefer using validated bboxes from filtering stage (ensures consistency)
                    detected_objects = []
                    
                    if validated_bboxes:
                        # Use pre-validated bboxes - these are guaranteed to match objects that passed filtering
                        for obj_name in group_objects:
                            if obj_name in validated_bboxes:
                                bbox_info = validated_bboxes[obj_name]
                                detected_objects.append({
                                    'bbox': bbox_info['bbox'],
                                    'class_name': obj_name
                                })
                            else:
                                logger.debug(f"Object {obj_name} not in validated_bboxes (may have been filtered out)")
                    else:
                        # Fallback: re-extract and match bboxes (less reliable, may have inconsistencies)
                        all_bboxes = VisualizationUtils.extract_bboxes_from_segmentation(str(seg_image_path))
                        
                        # Build mapping: color_int -> bbox (for fast lookup)
                        color_to_bbox = {}
                        for bbox_info in all_bboxes:
                            color_to_bbox[bbox_info['segment_id']] = bbox_info
                        
                        # Match group objects to bboxes (objects are already validated in filtering)
                        for obj_name in group_objects:
                            sm_names = taxonomy_to_sm_names.get(obj_name, [])
                            
                            # Track best (largest visible area) match among duplicates
                            best_candidate = None
                            best_area = -1.0
                            
                            for sm_name in sm_names:
                                # Get color for this SM name from seenable_obj_dict
                                color = seenable_obj_data.get(sm_name)
                                if color and isinstance(color, list) and len(color) >= 3:
                                    color_int = color[0] * 256 * 256 + color[1] * 256 + color[2]
                                    
                                    # Look up bbox by color (bboxes are already validated in filtering)
                                    bbox_info = color_to_bbox.get(color_int)
                                    if bbox_info:
                                        x1, y1, x2, y2 = bbox_info['bbox'][:4]
                                        width = max(0, x2 - x1)
                                        height = max(0, y2 - y1)
                                        area = float(width * height)
                                        
                                        # Track best candidate (largest area)
                                        if area > best_area:
                                            best_area = area
                                            best_candidate = bbox_info
                            
                            # All objects should have bboxes since they passed filtering
                            # But handle edge case where filtering happened but bbox lookup fails
                            if best_candidate is not None:
                                detected_objects.append({
                                    'bbox': best_candidate['bbox'],
                                    'class_name': obj_name
                                })
                            else:
                                logger.warning(f"No bbox found for {obj_name} in segmentation (tried SM names: {sm_names}) - object may have been filtered incorrectly")
                
                    # Create bbox visualization
                    if detected_objects:
                        detected_objects = ImageProcessingUtils._filter_overlapping_detections(
                            detected_objects,
                            iou_threshold=0.2,
                            coverage_threshold=0.85,
                            scene_id=scene_id,
                            group_idx=group_idx
                        )
                    
                    if detected_objects:
                        # Save sidecar with exact objects used for drawing and their box colors
                        try:
                            sidecar = group_output_dir / "objects_used.json"
                            # Color names by index (matching VisualizationUtils.draw_2d_bbox_image exactly)
                            color_names = [
                                "red", "green", "blue", "yellow", "orange", "pink", "purple",
                                "magenta", "cyan", "rose", "violet", "turquoise"
                            ]
                            
                            # Create mapping: color box -> object name
                            objects_with_boxes = {}
                            for idx, obj in enumerate(detected_objects):
                                color_name = color_names[idx % len(color_names)]
                                box_label = f"{color_name.capitalize()} Box"
                                objects_with_boxes[box_label] = obj['class_name']
                            
                            # Save both formats: box mapping and simple list
                            output_data = {
                                "box_mapping": objects_with_boxes,
                                "objects": [obj['class_name'] for obj in detected_objects]
                            }
                            with open(sidecar, 'w') as f:
                                json.dump(output_data, f, indent=2)
                        except Exception:
                            pass
                        VisualizationUtils.draw_2d_bbox_image(str(lit_image_path), detected_objects, target_bbox)
                        logger.info(f"Created bbox image for {scene_id}_group{group_idx} with {len(detected_objects)} objects")
                        
                        # Return list of objects that have bboxes
                        objects_with_bboxes = [obj['class_name'] for obj in detected_objects]
                        return objects_with_bboxes
                    else:
                        # Fallback: copy resized image
                        VisualizationUtils.resize_and_save_image(str(lit_image_path), target_bbox)
                        logger.warning(f"No bboxes found for group, copied resized image")
                        return []
                elif not taxonomy_to_sm_names:
                    # No taxonomy_to_sm_names mapping available
                    VisualizationUtils.resize_and_save_image(str(lit_image_path), target_bbox)
                    logger.warning(f"No taxonomy_to_sm_names mapping for {scene_id}_group{group_idx}, using resized image")
                elif not seenable_obj_data:
                    # No seenable_obj_dict available
                    VisualizationUtils.resize_and_save_image(str(lit_image_path), target_bbox)
                    logger.warning(f"No seenable_obj_dict for {scene_id}_group{group_idx}, using resized image")
            else:
                # No segmentation - fallback to resized image
                VisualizationUtils.resize_and_save_image(str(lit_image_path), target_bbox)
                logger.warning(f"No segmentation for {scene_id}, using resized image")
            
        except Exception as e:
            logger.error(f"Error copying images for {scene_id}_group{group_idx}: {e}")
            # Fallback: ensure bbox.jpg exists even if there was an error
            target_bbox = group_output_dir / "bbox.jpg"
            if lit_image_path.exists() and not target_bbox.exists():
                VisualizationUtils.resize_and_save_image(str(lit_image_path), target_bbox)
                logger.warning(f"Created fallback bbox.jpg due to error")
        
        return []
    
    @staticmethod
    def _filter_overlapping_detections(
        detected_objects: List[Dict[str, Any]],
        iou_threshold: float,
        coverage_threshold: float,
        scene_id: str,
        group_idx: int,
    ) -> List[Dict[str, Any]]:
        """Remove objects whose bboxes are almost entirely covered by a larger bbox."""
        if len(detected_objects) < 2:
            return detected_objects
        
        def _area(bbox: List[int]) -> int:
            if not bbox or len(bbox) < 4:
                return 0
            x1, y1, x2, y2 = bbox[:4]
            return max(0, x2 - x1) * max(0, y2 - y1)
        
        def _intersection_and_iou(bbox_a: List[int], bbox_b: List[int]) -> (int, float):
            if len(bbox_a) < 4 or len(bbox_b) < 4:
                return 0, 0.0
            ax1, ay1, ax2, ay2 = bbox_a[:4]
            bx1, by1, bx2, by2 = bbox_b[:4]
            
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            
            if ix2 <= ix1 or iy2 <= iy1:
                return 0, 0.0
            
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            area_a = _area(bbox_a)
            area_b = _area(bbox_b)
            union = area_a + area_b - inter_area
            iou = inter_area / union if union > 0 else 0.0
            return inter_area, iou
        
        keep = [True] * len(detected_objects)
        removed_pairs = []
        
        for i in range(len(detected_objects)):
            if not keep[i]:
                continue
            bbox_i = detected_objects[i].get('bbox', [])
            area_i = _area(bbox_i)
            if area_i == 0:
                keep[i] = False
                removed_pairs.append((detected_objects[i].get('class_name'), None, 0.0, 0.0))
                continue
            for j in range(len(detected_objects)):
                if i == j or not keep[j]:
                    continue
                bbox_j = detected_objects[j].get('bbox', [])
                inter_area, iou = _intersection_and_iou(bbox_i, bbox_j)
                if iou < iou_threshold:
                    continue
                
                area_j = _area(bbox_j)
                if area_j == 0:
                    keep[j] = False
                    removed_pairs.append((detected_objects[j].get('class_name'), detected_objects[i].get('class_name'), 0.0, iou))
                    continue
                
                if area_i <= area_j:
                    smaller_idx, larger_idx = i, j
                    smaller_area = area_i
                else:
                    smaller_idx, larger_idx = j, i
                    smaller_area = area_j
                
                if smaller_area == 0:
                    continue
                
                coverage = inter_area / smaller_area
                if coverage >= coverage_threshold:
                    keep[smaller_idx] = False
                    removed_pairs.append((
                        detected_objects[smaller_idx].get('class_name'),
                        detected_objects[larger_idx].get('class_name'),
                        coverage,
                        iou
                    ))
                    if smaller_idx == i:
                        break
            if not keep[i]:
                continue
        
        if removed_pairs:
            for removed_name, kept_name, coverage, iou in removed_pairs:
                if removed_name:
                    if kept_name:
                        logger.info(
                            f"Removing nested bbox for {removed_name} in {scene_id}_group{group_idx}: "
                            f"covered {coverage:.2f} (IoU {iou:.2f}) by {kept_name}"
                        )
                    else:
                        logger.info(
                            f"Removing invalid bbox for {removed_name} in {scene_id}_group{group_idx}"
                        )
        
        return [obj for idx, obj in enumerate(detected_objects) if keep[idx]]
    
    @staticmethod
    def _load_color_mappings(source_dir: Path, group_objects: List[str]) -> Dict[str, List[List[int]]]:
        """Load color mappings for group objects from seenable_obj_dict.json.
        
        Returns a dictionary mapping taxonomy names to lists of [R, G, B] colors.
        """
        obj_color_map = {}
        group_objects_set = set(group_objects)
        
        # Load SM to taxonomy mapping
        all_objects_file = Path(__file__).parent.parent / "sim_scene_object" / "data" / "object_list_v4.json"
        sm_to_taxonomy = {}
        
        try:
            if all_objects_file.exists():
                with open(all_objects_file, 'r') as f:
                    all_objects_data = json.load(f)
                for taxonomy_name, entries in all_objects_data.items():
                    for entry in entries:
                        sm_name = entry.get('object name', '')
                        if sm_name:
                            sm_to_taxonomy[sm_name] = taxonomy_name
        except Exception as e:
            logger.warning(f"Could not load all_objects mapping: {e}")
        
        # Load seenable_obj_dict.json
        seenable_obj_file = source_dir / "seenable_obj_dict.json"
        
        try:
            if not seenable_obj_file.exists():
                logger.warning(f"No seenable_obj_dict.json found for {source_dir.name}")
                return {}
            
            with open(seenable_obj_file, 'r') as f:
                seenable_obj_data = json.load(f)
            
            # Build mapping: taxonomy_name -> colors (exact match only)
            for sm_name, color in seenable_obj_data.items():
                if not isinstance(color, list) or len(color) < 3:
                    continue
                
                # Direct lookup taxonomy from sm_to_taxonomy (exact match only)
                taxonomy_name = sm_to_taxonomy.get(sm_name)
                
                # If this SM object maps to a taxonomy name in group_objects, add it
                if taxonomy_name and taxonomy_name in group_objects_set:
                    if taxonomy_name not in obj_color_map:
                        obj_color_map[taxonomy_name] = []
                    obj_color_map[taxonomy_name].append(color)
            
            matched_count = sum(1 for obj in group_objects if obj in obj_color_map)
            logger.info(f"Matched {matched_count}/{len(group_objects)} objects to colors")
            
            return obj_color_map
            
        except Exception as e:
            logger.error(f"Could not load color mappings: {e}")
            return {}
    
    @staticmethod
    def get_object_detections_for_image(image_id: str, source_dir: Path) -> List[Dict]:
        """Get object detections with bounding boxes for visualization"""
        try:
            # Look for annotation files in the source directory
            annotation_files = []
            
            # Check for annotations subdirectory
            annotations_dir = source_dir / "annotations"
            if annotations_dir.exists():
                # Look for refined annotations first, then regular annotations
                refined_files = list(annotations_dir.glob("*_refined.json"))
                if refined_files:
                    annotation_files.extend(refined_files)
                else:
                    json_files = list(annotations_dir.glob("*.json"))
                    json_files = [f for f in json_files if not f.name.endswith("_refined.json")]
                    annotation_files.extend(json_files)
            
            # If no annotations subdirectory, look for JSON files directly in source directory
            if not annotation_files:
                json_files = list(source_dir.glob("*.json"))
                annotation_files.extend(json_files)
            
            if not annotation_files:
                logger.warning(f"No annotation files found for {image_id}")
                return []
            
            # Load the first annotation file found
            annotation_file = annotation_files[0]
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)
            
            # Handle different annotation formats
            if 'detections' in annotation_data:
                detections = annotation_data['detections']
            elif 'detected_objects' in annotation_data:
                detections = annotation_data['detected_objects']
            else:
                logger.warning(f"No detections found in annotation file for {image_id}")
                return []
            
            # Extract object data with bounding boxes
            detected_objects = []
            for detection in detections:
                if isinstance(detection, dict):
                    class_name = detection.get('class_name', detection.get('class', 'unknown'))
                    # Try different bbox field names
                    bbox = detection.get('xyxy', detection.get('bbox', detection.get('bounding_box', [])))
                    
                    if class_name and class_name != 'unknown' and bbox and len(bbox) >= 4:
                        detected_objects.append({
                            'bbox': bbox,
                            'class_name': class_name
                        })
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error getting object detections for {image_id}: {e}")
            return []
