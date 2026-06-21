#!/usr/bin/env python3
"""
Restore bbox images to original resolution from annotations.
For sim images: uses annotations from /path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations
For real images: uses openimages_unified_output
"""

import json
import sys
import re
from pathlib import Path
from PIL import Image
import logging

sys.path.insert(0, str(Path(__file__).parent / "modules" / "qa_modules"))
from visualization_utils import VisualizationUtils
from data_loading_utils import DataLoadingUtils
from annotation_processing_utils import AnnotationProcessingUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def restore_simimage_bbox_images():
    """Restore simimage bbox images to original resolution from annotations"""
    annotations_base = Path("/path/to/Taxonomy/Data/SimulationMetadata/scenes/annotations")
    simimage_final = Path("taxonomy_datagen/SpatialReasonerDataGen/qa_gen/taxonomyQABench_simimage_final")
    images_dir = simimage_final / "images"
    questions_file = simimage_final / "all_questions.json"
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return
    
    # Load questions to get objects used per image_id
    questions_by_image = {}
    if questions_file.exists():
        with open(questions_file, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
        for q in all_questions:
            image_id = q.get('image_id', '')
            if image_id:
                if image_id not in questions_by_image:
                    questions_by_image[image_id] = []
                questions_by_image[image_id].append(q)
        logger.info(f"Loaded questions for {len(questions_by_image)} image IDs")
    else:
        logger.warning(f"Questions file not found: {questions_file}")
    
    logger.info(f"Processing simimage bbox images in {images_dir}")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for image_id_dir in sorted(images_dir.iterdir()):
        if not image_id_dir.is_dir():
            continue
        
        image_id = image_id_dir.name
        bbox_file = image_id_dir / "bbox.jpg"
        
        if not bbox_file.exists():
            logger.debug(f"Skipping {image_id}: no bbox.jpg")
            skipped += 1
            continue
        
        try:
            # Parse image_id: e.g., "1940Office_l057_r001_group0" -> scene="1940Office", view="l057_r001"
            # Special case: "ConferenceRoomVol1_1_combo0_0_l014_r005_group0" -> scene="ConferenceRoomVol1_1", view="combo0_0_l014_r005"
            image_id_no_group = image_id.rsplit("_group", 1)[0]
            parts = image_id_no_group.split("_")
            
            # Find index where "l" pattern starts (e.g., "l057")
            view_start_idx = None
            for i, part in enumerate(parts):
                if part.startswith("l") and len(part) > 1 and part[1].isdigit():
                    view_start_idx = i
                    break
            
            if view_start_idx is None:
                logger.warning(f"Could not parse view_id from {image_id}")
                skipped += 1
                continue
            
            # Check if there's a number immediately before "l" that should be part of view_id
            # e.g., "JapaneseRestaurant_02_1_l037_r005" -> scene="JapaneseRestaurant_02", view="1_l037_r005"
            if view_start_idx > 0 and parts[view_start_idx - 1].isdigit():
                scene_name = "_".join(parts[:view_start_idx - 1])
                view_id = "_".join(parts[view_start_idx - 1:])
            else:
                scene_name = "_".join(parts[:view_start_idx])
                view_id = "_".join(parts[view_start_idx:])
            
            # Check if scene_name contains "combo" - if so, it should be part of view_id
            # e.g., "ConferenceRoomVol1_1_combo0_0" -> scene="ConferenceRoomVol1_1", view="combo0_0_l014_r005"
            if "_combo" in scene_name:
                combo_parts = scene_name.split("_combo", 1)
                if len(combo_parts) == 2:
                    scene_name = combo_parts[0]
                    combo_part = "combo" + combo_parts[1]
                    view_id = combo_part + "_" + view_id
            
            annotation_dir = annotations_base / scene_name / view_id
            annotation_json = annotation_dir / "scene_annotations_split.json"
            
            if not annotation_json.exists():
                logger.warning(f"Annotation not found for {image_id}: {annotation_json}")
                skipped += 1
                continue
            
            with open(annotation_json, 'r') as f:
                annotation_data = json.load(f)
            
            original_image_path = Path(annotation_data.get('image_path', ''))
            if not original_image_path.exists():
                logger.warning(f"Original image not found: {original_image_path}")
                skipped += 1
                continue
            
            original_img = Image.open(original_image_path)
            original_width, original_height = original_img.size
            logger.info(f"{image_id}: Original resolution: {original_width}x{original_height}")
            
            # Get objects used in questions for this image_id
            questions_for_image = questions_by_image.get(image_id, [])
            
            # Extract box_to_object mapping from questions to determine color assignment
            box_to_object_mapping = None
            for q in questions_for_image:
                if 'box_to_object' in q and q['box_to_object']:
                    box_to_object_mapping = q['box_to_object']
                    break
            
            # Color name to index mapping (matching VisualizationUtils.draw_2d_bbox_image)
            color_name_to_index = {
                "red": 0, "green": 1, "blue": 2, "yellow": 3, "orange": 4, 
                "pink": 5, "purple": 6, "magenta": 7, "cyan": 8, 
                "rose": 9, "violet": 10, "turquoise": 11
            }
            
            # Create object -> color_index mapping
            object_to_color_index = {}
            if box_to_object_mapping:
                for color_box, obj_name in box_to_object_mapping.items():
                    color_name = color_box.replace(" box", "").lower()
                    if color_name in color_name_to_index:
                        object_to_color_index[obj_name.lower()] = color_name_to_index[color_name]
            
            # First try to get from objects_used.json (most reliable)
            objects_used_file = image_id_dir / "objects_used.json"
            objects_order = []
            if objects_used_file.exists():
                with open(objects_used_file, 'r') as f:
                    objects_used_data = json.load(f)
                objects_order = objects_used_data.get('objects', [])
                logger.debug(f"{image_id}: Got objects from objects_used.json: {objects_order}")
            
            # Fallback to questions if objects_used.json doesn't have objects
            if not objects_order:
                objects_in_questions = set()
                
                for q in questions_for_image:
                    # Collect objects from various fields
                    for obj_list in [q.get('objects', []), q.get('choices', [])]:
                        for obj in obj_list:
                            if obj and obj not in objects_in_questions:
                                objects_in_questions.add(obj)
                                objects_order.append(obj)
                    # Also check box_to_object values
                    if 'box_to_object' in q:
                        for obj in q['box_to_object'].values():
                            if obj and obj not in objects_in_questions:
                                objects_in_questions.add(obj)
                                objects_order.append(obj)
            
            if not objects_order:
                logger.warning(f"No objects found for {image_id}, skipping")
                skipped += 1
                continue
            
            # Sort objects by color index if we have the mapping
            if object_to_color_index:
                objects_order.sort(key=lambda obj: object_to_color_index.get(obj.lower(), 999))
                logger.info(f"{image_id}: Objects ordered by color: {objects_order}")
            else:
                logger.info(f"{image_id}: Objects to include: {objects_order} (no color mapping found)")
            
            objects_in_questions = set(obj.lower() for obj in objects_order)
            
            foreground = annotation_data.get('foreground', {})
            all_detected_objects = {}  # Map taxonomy_name -> detection info
            
            # Load SM to taxonomy mapping using DataLoadingUtils (same as generation code)
            data_loader = DataLoadingUtils()
            sm_to_taxonomy = data_loader.load_sm_to_human_mapping()
            
            # Collect all objects from annotations
            # Group by taxonomy name (since multiple SM objects can map to same taxonomy name)
            # Same as generation: we'll select the one with largest bbox area
            taxonomy_to_instances = {}  # taxonomy_name_lower -> list of detection info
            
            for category, category_objects in foreground.items():
                for obj in category_objects:
                    sm_name = obj.get('object_id')
                    bbox_2d = obj.get('bbox_2d')
                    if sm_name and bbox_2d and len(bbox_2d) >= 4:
                        taxonomy_name = sm_to_taxonomy.get(sm_name, sm_name)
                        taxonomy_name_lower = taxonomy_name.lower()
                        
                        # Calculate bbox area (same as generation: selects largest area)
                        x1, y1, x2, y2 = bbox_2d[:4]
                        width = max(0, x2 - x1)
                        height = max(0, y2 - y1)
                        area = width * height
                        
                        if taxonomy_name_lower not in taxonomy_to_instances:
                            taxonomy_to_instances[taxonomy_name_lower] = []
                        
                        taxonomy_to_instances[taxonomy_name_lower].append({
                            'bbox': bbox_2d,
                            'class_name': taxonomy_name,
                            'original_name': taxonomy_name,
                            'sm_name': sm_name,
                            'area': area
                        })
            
            # Filter to only include objects that are in questions, preserving order
            # For each object, select the instance with largest bbox area (same as generation)
            detected_objects = []
            used_bboxes = set()  # Track used bboxes to avoid duplicates
            
            for obj_name in objects_order:
                obj_name_lower = obj_name.lower()
                if obj_name_lower in taxonomy_to_instances:
                    instances = taxonomy_to_instances[obj_name_lower]
                    
                    # Filter out already used bboxes
                    available_instances = []
                    for inst in instances:
                        bbox_key = tuple(inst['bbox'][:4])
                        if bbox_key not in used_bboxes:
                            available_instances.append(inst)
                    
                    if available_instances:
                        # Select the one with largest area (same as generation pipeline)
                        best_instance = max(available_instances, key=lambda x: x['area'])
                        detected_objects.append({
                            'bbox': best_instance['bbox'],
                            'class_name': best_instance['class_name']
                        })
                        used_bboxes.add(tuple(best_instance['bbox'][:4]))
                        logger.debug(f"{image_id}: Selected '{obj_name}' instance with area {best_instance['area']:.0f} (from {len(instances)} instances)")
                    else:
                        logger.warning(f"Object '{obj_name}' from questions: all instances already used for {image_id}")
                else:
                    logger.warning(f"Object '{obj_name}' from questions not found in annotations for {image_id}")
            
            if not detected_objects:
                logger.warning(f"No matching bboxes found for objects in questions for {image_id}")
                skipped += 1
                continue
            
            logger.info(f"{image_id}: Drawing {len(detected_objects)} bboxes (filtered from {len(all_detected_objects)} total) at original resolution")
            VisualizationUtils.draw_2d_bbox_image(
                str(original_image_path),
                detected_objects,
                bbox_file,
                target_width=original_width,
                target_height=original_height
            )
            
            # Also restore original.jpg to original size
            original_jpg_file = image_id_dir / "original.jpg"
            if original_jpg_file.exists():
                original_img = Image.open(original_image_path)
                # Convert to RGB if necessary (JPEG doesn't support RGBA)
                if original_img.mode == 'RGBA':
                    rgb_img = Image.new('RGB', original_img.size, (255, 255, 255))
                    rgb_img.paste(original_img, mask=original_img.split()[3])
                    original_img = rgb_img
                elif original_img.mode != 'RGB':
                    original_img = original_img.convert('RGB')
                original_img.save(original_jpg_file, quality=100)
                logger.debug(f"{image_id}: Restored original.jpg to {original_width}x{original_height}")
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {processed} images...")
        
        except Exception as e:
            logger.error(f"Error processing {image_id}: {e}")
            errors += 1
    
    logger.info(f"Simimage restoration complete: {processed} processed, {skipped} skipped, {errors} errors")


def restore_realimage_bbox_images():
    """Restore realimage bbox images to original resolution from openimages_unified_output"""
    openimages_output = Path("taxonomy_datagen/SpatialReasonerDataGen/qa_gen/openimages_unified_output")
    realimage_final = Path("taxonomy_datagen/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_final_polished")
    images_dir = realimage_final / "images"
    original_images_dir = Path("/path/to/project/openimages_train_10000")
    questions_file = realimage_final / "all_questions.json"
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return
    
    if not original_images_dir.exists():
        logger.error(f"Original images directory not found: {original_images_dir}")
        return
    
    # Load questions to get objects used per image_id
    questions_by_image = {}
    if questions_file.exists():
        with open(questions_file, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
        for q in all_questions:
            image_id = q.get('image_id', '')
            if not image_id:
                # Try to extract from image_path
                image_path = q.get('image_path', '')
                if image_path:
                    image_id = image_path.split('/')[0]
            if image_id:
                if image_id not in questions_by_image:
                    questions_by_image[image_id] = []
                questions_by_image[image_id].append(q)
        logger.info(f"Loaded questions for {len(questions_by_image)} image IDs")
    else:
        logger.warning(f"Questions file not found: {questions_file}")
    
    logger.info(f"Processing realimage bbox images in {images_dir}")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for image_id_dir in sorted(images_dir.iterdir()):
        if not image_id_dir.is_dir():
            continue
        
        image_id = image_id_dir.name
        bbox_file = image_id_dir / "bbox.jpg"
        
        if not bbox_file.exists():
            logger.debug(f"Skipping {image_id}: no bbox.jpg")
            skipped += 1
            continue
        
        try:
            # Try to find annotations in openimages_unified_output first
            annotation_dir = openimages_output / image_id
            annotation_json = None
            if annotation_dir.exists():
                # Check annotations subdirectory first (preferred location)
                annotations_subdir = annotation_dir / "annotations"
                if annotations_subdir.exists():
                    # Prefer refined annotations, then regular annotations
                    refined_json = annotations_subdir / f"{image_id}_refined.json"
                    if refined_json.exists():
                        annotation_json = refined_json
                    else:
                        regular_json = annotations_subdir / f"{image_id}.json"
                        if regular_json.exists():
                            annotation_json = regular_json
                        else:
                            # Fallback: any JSON file in annotations subdirectory
                            json_files = list(annotations_subdir.glob("*.json"))
                            if json_files:
                                annotation_json = json_files[0]
                
                # If not found in annotations subdirectory, try root of annotation_dir
                if not annotation_json or not annotation_json.exists():
                    annotation_json = annotation_dir / "annotations.json"
                    if not annotation_json.exists():
                        # Try other JSON files in root
                        json_files = list(annotation_dir.glob("*.json"))
                        if json_files:
                            annotation_json = json_files[0]
            
            # If still not found, try annotations subdirectory in image_dir
            if not annotation_json or not annotation_json.exists():
                annotations_subdir = image_id_dir / "annotations"
                if annotations_subdir.exists():
                    json_files = list(annotations_subdir.glob("*.json"))
                    if json_files:
                        annotation_json = json_files[0]
            
            # Find original image
            original_image_path = original_images_dir / f"{image_id}.jpg"
            if not original_image_path.exists():
                logger.warning(f"Original image not found for {image_id}: {original_image_path}")
                skipped += 1
                continue
            
            original_img = Image.open(original_image_path)
            original_width, original_height = original_img.size
            logger.info(f"{image_id}: Original resolution: {original_width}x{original_height}")
            
            # Get objects used in questions for this image_id
            questions_for_image = questions_by_image.get(image_id, [])
            
            # Extract box_to_object mapping from questions to determine color assignment
            # box_to_object maps: "Red box" -> "waterway", "Green box" -> "person", etc.
            # We need to reverse this to: "waterway" -> "Red box" (index 0), "person" -> "Green box" (index 1), etc.
            box_to_object_mapping = None
            for q in questions_for_image:
                if 'box_to_object' in q and q['box_to_object']:
                    box_to_object_mapping = q['box_to_object']
                    break
            
            # Color name to index mapping (matching VisualizationUtils.draw_2d_bbox_image)
            color_name_to_index = {
                "red": 0, "green": 1, "blue": 2, "yellow": 3, "orange": 4, 
                "pink": 5, "purple": 6, "magenta": 7, "cyan": 8, 
                "rose": 9, "violet": 10, "turquoise": 11
            }
            
            # Create object -> color_index mapping
            object_to_color_index = {}
            if box_to_object_mapping:
                for color_box, obj_name in box_to_object_mapping.items():
                    # Extract color name from "Red box", "Green box", etc.
                    color_name = color_box.replace(" box", "").lower()
                    if color_name in color_name_to_index:
                        object_to_color_index[obj_name.lower()] = color_name_to_index[color_name]
            
            # Get objects from questions - use the 'objects' field which matches available_objects from generation
            # This is the normalized order used during generation
            objects_order = []
            objects_seen = set()
            
            # Get objects from first question (they should all have the same objects list)
            if questions_for_image:
                first_q = questions_for_image[0]
                # The 'objects' field contains the normalized available_objects in order
                for obj in first_q.get('objects', []):
                    if obj and isinstance(obj, str) and obj.lower() not in objects_seen:
                        objects_seen.add(obj.lower())
                        objects_order.append(obj)
            
            if not objects_order:
                logger.warning(f"No objects found in questions for {image_id}, skipping")
                skipped += 1
                continue
            
            # Sort objects by color index if we have the mapping (to match box colors)
            if object_to_color_index:
                objects_order.sort(key=lambda obj: object_to_color_index.get(obj.lower(), 999))
                logger.info(f"{image_id}: Objects ordered by color: {objects_order}")
            else:
                logger.info(f"{image_id}: Objects from questions: {objects_order}")
            
            # Load annotations if available
            # Use the same logic as generation pipeline: normalize person classes and match by normalized class_name
            annotation_processor = AnnotationProcessingUtils()
            normalized_detections = []
            
            if annotation_json and annotation_json.exists():
                with open(annotation_json, 'r') as f:
                    annotation_data = json.load(f)
                
                # Handle different annotation formats (same as get_available_objects_for_image)
                detections = []
                if 'detections' in annotation_data:
                    detections = annotation_data['detections']
                elif 'detected_objects' in annotation_data:
                    detections = annotation_data['detected_objects']
                elif 'annotations' in annotation_data:
                    detections = annotation_data['annotations']
                elif isinstance(annotation_data, list):
                    detections = annotation_data
                
                # Normalize detections using the same logic as generation pipeline
                for detection in detections:
                    if isinstance(detection, dict):
                        class_name = detection.get('class_name', detection.get('class', 'unknown'))
                        if class_name and class_name != 'unknown':
                            # Normalize human-related classes to 'person' (same as generation)
                            normalized_class_name = annotation_processor.PERSON_NORMALIZATION.get(class_name.lower(), class_name)
                            
                            # Get bbox
                            bbox = detection.get('bbox', detection.get('xyxy', []))
                            if isinstance(bbox, str):
                                # Handle scientific notation: e.g., "5.9273950e+02" or "6.9171143e-01"
                                # Match numbers including scientific notation
                                numbers = re.findall(r'[\d.]+(?:[eE][+-]?\d+)?', bbox)
                                if len(numbers) >= 4:
                                    bbox = [float(n) for n in numbers[:4]]
                            
                            if bbox and len(bbox) >= 4:
                                x1, y1, x2, y2 = bbox[:4]
                                # Validate bbox
                                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                                    continue
                                
                                # Create normalized detection (same structure as generation)
                                detection_copy = detection.copy()
                                detection_copy['class_name'] = normalized_class_name
                                detection_copy['bbox'] = bbox[:4]
                                if 'class' in detection_copy:
                                    detection_copy['class'] = normalized_class_name
                                normalized_detections.append(detection_copy)
            
            # Filter by bbox coverage (same as generation pipeline)
            if normalized_detections:
                original_image_path = original_images_dir / f"{image_id}.jpg"
                if original_image_path.exists():
                    filtered_detections, _ = annotation_processor.filter_detections_by_coverage(
                        normalized_detections, original_image_path, max_coverage_ratio=0.6
                    )
                    normalized_detections = filtered_detections
            
            # Group detections by normalized class_name (same as generation matching logic)
            all_detected_objects = {}  # normalized_class_name -> list of detections
            for detection in normalized_detections:
                normalized_class_name = detection.get('class_name', '')
                if normalized_class_name:
                    if normalized_class_name not in all_detected_objects:
                        all_detected_objects[normalized_class_name] = []
                    all_detected_objects[normalized_class_name].append(detection)
            
            if not all_detected_objects:
                logger.warning(f"No bboxes found in annotations for {image_id}, skipping")
                skipped += 1
                continue
            
            # Match objects to detections using the SAME logic as generation pipeline
            # Generation matches: class_name == obj_name (both normalized)
            # Then randomly selects one detection per object type
            # We'll select by highest confidence to be deterministic
            import random
            
            detected_objects = []
            used_bboxes = set()  # Track which bboxes we've used
            
            for obj_name in objects_order:
                # Match based on normalized class_name (same as generation)
                # obj_name is already normalized (from questions 'objects' field)
                obj_name_normalized = obj_name.lower()
                
                if obj_name_normalized in all_detected_objects:
                    matching_detections = all_detected_objects[obj_name_normalized]
                    
                    # Filter out already used detections
                    available_detections = []
                    for det in matching_detections:
                        bbox_key = tuple(det.get('bbox', det.get('xyxy', [])))
                        if bbox_key not in used_bboxes:
                            available_detections.append(det)
                    
                    if available_detections:
                        # Select by highest confidence (deterministic alternative to random.choice)
                        selected_detection = max(available_detections, key=lambda d: d.get('confidence', 0.0))
                        
                        # Ensure class_name matches obj_name (same as generation)
                        selected_detection_copy = selected_detection.copy()
                        selected_detection_copy['class_name'] = obj_name
                        if 'class' in selected_detection_copy:
                            selected_detection_copy['class'] = obj_name
                        
                        detected_objects.append(selected_detection_copy)
                        used_bboxes.add(tuple(selected_detection.get('bbox', selected_detection.get('xyxy', []))))
                    else:
                        logger.warning(f"No available detection for '{obj_name}' in {image_id} (all already used)")
                else:
                    logger.warning(f"Object '{obj_name}' from questions not found in normalized detections for {image_id}")
            
            if not detected_objects:
                logger.warning(f"No matching bboxes found for objects in questions for {image_id}")
                skipped += 1
                continue
            
            logger.info(f"{image_id}: Drawing {len(detected_objects)} bboxes (filtered from {len(all_detected_objects)} total) at original resolution")
            VisualizationUtils.draw_2d_bbox_image(
                str(original_image_path),
                detected_objects,
                bbox_file,
                target_width=original_width,
                target_height=original_height
            )
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {processed} images...")
        
        except Exception as e:
            logger.error(f"Error processing {image_id}: {e}")
            errors += 1
    
    logger.info(f"Realimage restoration complete: {processed} processed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Restore bbox images to original resolution")
    parser.add_argument("--simimage", action="store_true", help="Restore simimage bbox images")
    parser.add_argument("--realimage", action="store_true", help="Restore realimage bbox images")
    parser.add_argument("--both", action="store_true", help="Restore both simimage and realimage")
    
    args = parser.parse_args()
    
    if args.both or (not args.simimage and not args.realimage):
        restore_simimage_bbox_images()
        restore_realimage_bbox_images()
    elif args.simimage:
        restore_simimage_bbox_images()
    elif args.realimage:
        restore_realimage_bbox_images()

