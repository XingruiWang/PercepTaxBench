#!/usr/bin/env python3
"""
Generate Taxonomy QA Benchmark for Real Images
This script generates comprehensive QA datasets from real images using unified QA generation.
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add modules to path
sys.path.append('/path/to/SpatialReasonerDataGen/qa_gen/scripts')

from modules.qa_modules.taxonomy_utils import TaxonomyUtils
from modules.qa_modules.cot_reasoning_utils import CoTReasoningGenerator
from modules.qa_modules.image_processing_utils import ImageProcessingUtils
from modules.qa_modules.object_utils import ObjectUtils
from modules.qa_modules.question_generation_utils import QuestionGenerationUtils
from modules.qa_modules.data_loading_utils import DataLoadingUtils
from modules.qa_modules.annotation_processing_utils import AnnotationProcessingUtils
from modules.qa_modules.unified_qa_generation_utils import UnifiedQAGenerationUtils
from modules.qa_modules.question_type_grouping import get_simplified_question_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxonomyQABenchGenerator:
    """Generate comprehensive QA benchmark from real images"""
    
    def __init__(self):
        """Initialize the generator with all required components"""
        # Initialize taxonomy utilities
        taxonomy_dir = Path('/path/to/SpatialReasonerDataGen/qa_gen/taxonomy')
        self.taxonomy_utils = TaxonomyUtils(taxonomy_dir)
        
        # Initialize modular components
        self.image_processing_utils = ImageProcessingUtils()
        self.object_utils = ObjectUtils(taxonomy_utils=self.taxonomy_utils)
        self.question_generation_utils = QuestionGenerationUtils(taxonomy_utils=self.taxonomy_utils)
        self.data_loading_utils = DataLoadingUtils()
        self.annotation_processing_utils = AnnotationProcessingUtils()
        # Set up annotations directory for spatial relationship calculation
        self.annotations_dir = Path('/path/to/SpatialReasonerDataGen/qa_gen/openimages_unified_output')
        
        self.unified_qa_generation_utils = UnifiedQAGenerationUtils(
            taxonomy_utils=self.taxonomy_utils,
            object_utils=self.object_utils,
            question_generation_utils=self.question_generation_utils,
            annotations_dir=self.annotations_dir,
            data_loading_utils=self.data_loading_utils
        )
        
        logger.info("Initialized Unified QA Generator with modular components")
    
    def process_images_with_qa_space(self, images_path: Path, output_dir: Path, 
                                   image_ids: List[str] = None, max_files: int = None, seed: int = 42):
        """Process images and generate QA using QA space analysis"""
        
        # Ensure random seed is set (defensive - seed should already be set in main())
        import random
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        logger.debug(f"Random seed set to {seed} in process_images_with_qa_space")
        
        # Load QA space data
        qa_space_data = self.data_loading_utils.load_qa_space_data()
        
        # Get list of images to process
        if image_ids:
            image_dirs = [images_path / img_id for img_id in image_ids if (images_path / img_id).exists()]
        else:
            # Sort to ensure deterministic processing order
            image_dirs = sorted(list(images_path.iterdir()), key=lambda x: x.name)
        if max_files:
            image_dirs = image_dirs[:max_files]
        
        logger.info(f"Processing {len(image_dirs)} images")
        
        all_questions = []
        all_scene_statistics = []
        processed_count = 0
        
        for image_dir in image_dirs:
            if not image_dir.is_dir():
                    continue
                
            image_id = image_dir.name
            logger.info(f"Processing {image_id}")
            
            # Get available objects for this image
            available_objects, detections = self.annotation_processing_utils.get_available_objects_for_image(
                image_id, qa_space_data, image_dir
            )
            
            logger.info(f"Found {len(available_objects)} objects for {image_id}: {available_objects}")
            
            if len(available_objects) < 3:
                logger.warning(f"Skipping {image_id}: only {len(available_objects)} objects (need at least 3)")
                continue
            
            if len(available_objects) > 6:
                logger.warning(f"Skipping {image_id}: {len(available_objects)} objects (need at most 6, too many for clear questions)")
                continue
        
            # Filter detections to match available_objects order (randomly select one per unique object type)
            # This ensures bbox colors match the order used in questions and number of boxes matches unique objects
            # available_objects contains normalized names (e.g., "person"), but detections may have original names (e.g., "woman")
            # need to match based on normalized class names
            ordered_detections = []
            for obj_name in available_objects:
                # Collect ALL detections for this object type (match normalized class names)
                matching_detections = []
        for detection in detections:
            if isinstance(detection, dict):
                        # Get normalized class name (should already be normalized by get_available_objects_for_image)
                        class_name = detection.get('class_name', detection.get('class', ''))
            elif isinstance(detection, str):
                        class_name = detection
            else:
                continue
                    
                # Match based on normalized class name
                if class_name == obj_name:
                    matching_detections.append(detection)
                
                # Randomly select one detection for this object type
                if matching_detections:
                    selected_detection = random.choice(matching_detections)
                    # Ensure the detection has normalized class_name for consistency
                    if isinstance(selected_detection, dict):
                        selected_detection_copy = selected_detection.copy()
                        # Ensure class_name matches the normalized obj_name
                        selected_detection_copy['class_name'] = obj_name
                        if 'class' in selected_detection_copy:
                            selected_detection_copy['class'] = obj_name
                        ordered_detections.append(selected_detection_copy)
                    else:
                        ordered_detections.append(selected_detection)
                else:
                    logger.warning(f"Warning: No detection found for object '{obj_name}' in {image_id}")
            
            if len(ordered_detections) != len(available_objects):
                logger.warning(f"Warning: Ordered detections count {len(ordered_detections)} != available_objects count {len(available_objects)} for {image_id}")
                # Skip this image if we can't match detections to available objects
                continue
            
            # Generate QA from QA space (already includes additional questions with spatial limiting)
            # Pass available_objects which will be used for choices (ensures order matches bbox colors)
            scene_questions = self.unified_qa_generation_utils.generate_qa_from_space(
                image_id, available_objects, qa_space_data
            )
            
            # Only copy images if questions were generated (don't save images with no QA)
            if scene_questions:
                # Copy images with bounding boxes (use ordered_detections to ensure color order matches)
                images_dir = output_dir / "images"
                images_dir.mkdir(exist_ok=True)
                original_images_dir = Path('/path/to/project/openimages_train_10000')
                self.image_processing_utils.copy_images_with_bbox(image_id, image_dir, images_dir, ordered_detections, original_images_dir)
                
                # Extract color assignments from detections (saved by draw_2d_bbox_image)
                # This is the ACTUAL color used when drawing - guaranteed to match the image
                detection_color_mapping = {}  # object_name -> color_name
                for detection in ordered_detections:
                    if isinstance(detection, dict):
                        class_name = detection.get('class_name', detection.get('class', ''))
                        color_name = detection.get('_bbox_color_name')
                        if class_name and color_name:
                            detection_color_mapping[class_name] = color_name
                
                # Store detection_color_mapping in all questions for this image
                # This will be used later to create box_to_object mapping from ACTUAL colors
                for question in scene_questions:
                    question['_detection_color_mapping'] = detection_color_mapping
            else:
                logger.info(f"Skipping image copy for {image_id}: no questions generated")
                continue
            
            # Strategy 5: Optional cap total questions at 15 per image (prioritize hard questions)
            original_question_count = len(scene_questions)
            if len(scene_questions) > 15:
                hard_question_types = ['repurposing_', 'counterfactual_', 'compositional_', 'latent_']
                hard_questions = [q for q in scene_questions 
                                if any(hard_type in q.get('question_type', '') for hard_type in hard_question_types)]
                easy_questions = [q for q in scene_questions 
                               if not any(hard_type in q.get('question_type', '') for hard_type in hard_question_types)]
                
                # Prioritize: keep all hard questions, then fill remaining slots with easy questions
                max_easy = max(0, 15 - len(hard_questions))
                selected_easy = random.sample(easy_questions, min(max_easy, len(easy_questions))) if easy_questions else []
                scene_questions = hard_questions + selected_easy
                logger.info(f"Capped questions from {original_question_count} to {len(scene_questions)} (kept {len(hard_questions)} hard, {len(selected_easy)} easy)")
            
            if scene_questions:
                # Ensure all questions have choices in the correct order (matching available_objects)
                for question in scene_questions:
                    # Force choices to match available_objects order (ensures color consistency)
                    question['choices'] = available_objects.copy()
                    question['objects'] = available_objects.copy()  # Also ensure objects match
                
                all_questions.extend(scene_questions)
                all_scene_statistics.append({
                                "image_id": image_id,
                    "objects": available_objects,
                    "questions_generated": len(scene_questions)
                })
                processed_count += 1
                    
            logger.info(f"Generated {len(scene_questions)} questions for {image_id}")
        
        # Add dual format for questions
        questions_to_remove = []
        for idx, question in enumerate(all_questions):
            question['question_index'] = idx
            
            # Convert question_type to question_category for JSON output
            detailed_type = question.pop('question_type', None)
            if detailed_type:
                question['question_category'] = get_simplified_question_type(detailed_type)
            
            # Add dual format: question/answer (with colored boxes) and original_question/original_answer (with object names)
            # CRITICAL: Use ACTUAL colors from bbox drawing, not inferred colors
            objects = question.get('choices', question.get('objects', []))
            
            # Keep choices and objects for backward compatibility, but box_to_object mapping is the source of truth
            if objects:
                # Use ACTUAL color assignments from bbox drawing (saved in detection_color_mapping)
                detection_color_mapping = question.pop('_detection_color_mapping', {})
                
                if detection_color_mapping:
                    # Create box_to_object mapping from ACTUAL colors used when drawing bboxes
                    # This is guaranteed to match the image
                    box_to_object = {}
                    obj_to_colored_box = {}
                    
                    for obj in objects:
                        # Get actual color name from detection
                        color_name = detection_color_mapping.get(obj)
                        if color_name:
                            # Capitalize first letter of each word
                            color_name_cap = ' '.join(word.capitalize() for word in color_name.split())
                            colored_box = f"{color_name_cap} box"
                            box_to_object[colored_box] = obj
                            obj_to_colored_box[obj] = colored_box
                        else:
                            # Fallback if color mapping not found (shouldn't happen)
                            logger.warning(f"Color mapping not found for object '{obj}', using index-based inference")
                            # Must match visualization_utils.py color_names exactly (prioritized order)
                            color_names = ["red", "green", "blue", "yellow", "orange", "pink", "purple", "magenta", "cyan", "rose", "violet", "turquoise"]
                            obj_idx = objects.index(obj) if obj in objects else 0
                            color_name_cap = color_names[obj_idx % len(color_names)].capitalize()
                            colored_box = f"{color_name_cap} box"
                            box_to_object[colored_box] = obj
                            obj_to_colored_box[obj] = colored_box
                    
                    question['box_to_object'] = box_to_object
                else:
                    # Fallback: infer colors from order (for backward compatibility or if mapping missing)
                    logger.warning(f"Detection color mapping not available for question {question.get('question_index', 'unknown')}, using index-based inference")
                    # Must match visualization_utils.py color_names exactly (prioritized order)
                    color_names = ["red", "green", "blue", "yellow", "orange", "pink", "purple", "magenta", "cyan", "rose", "violet", "turquoise"]
                    obj_to_colored_box = {obj: f"{color_names[i % len(color_names)].capitalize()} box" for i, obj in enumerate(objects)}
                    box_to_object = {colored_box: obj for obj, colored_box in obj_to_colored_box.items()}
                    question['box_to_object'] = box_to_object
                
                # Get original question and answer
                orig_question = question.get('question', '')
                orig_answer = question.get('answer', '')
                
                # Normalize "Option objects:" to "Objects to choose from:" format for consistency
                # Also extract the options list if present to avoid replacing object names in it
                options_suffix = ""
                question_text_only = orig_question
                if "Option objects:" in question_text_only:
                    parts = question_text_only.split("Option objects:", 1)
                    question_text_only = parts[0].strip()
                    options_suffix = "Option objects:" + parts[1] if len(parts) > 1 else ""
                elif "Objects to choose from:" in question_text_only:
                    parts = question_text_only.split("Objects to choose from:", 1)
                    question_text_only = parts[0].strip()
                    options_suffix = "Objects to choose from:" + parts[1] if len(parts) > 1 else ""
                
                # Create colored-box-based question and answer
                question_with_colored_boxes = question_text_only
                answer_with_colored_boxes = orig_answer
                
                # Replace answer with colored box if it's an object name
                if orig_answer in obj_to_colored_box:
                    answer_with_colored_boxes = obj_to_colored_box[orig_answer]
                
                # Check if this is a spatial question - need "object in" prefix
                is_spatial = question.get('question_category', '').startswith('spatial')
                
                # Extract affordance/function/material text to preserve it from replacement
                # Pattern: "Which object has the affordance of {text}?"
                affordance_match = re.search(r'has the (affordance|function|material) of ([^?]+)', question_with_colored_boxes)
                protected_text = []
                if affordance_match:
                    protected_text = affordance_match.group(2).strip().split()
                
                # Also protect object names that appear after "is" as part of descriptive phrases
                # BUT skip this protection for spatial questions - they need object names replaced
                # Patterns like "is prepared food", "is furniture", "is food", "is a device", etc.
                # Match: "is [optional adjective] {object_name}" or "is {object_name}"
                # This prevents replacing descriptive phrases like "prepared food" with "prepared Yellow box"
                if not is_spatial:
                    is_pattern_match = re.search(r'is\s+(?:prepared|used|a|an|for)?\s*([a-z]+(?:\s+[a-z]+)*)', question_with_colored_boxes.lower())
                    if is_pattern_match:
                        descriptive_phrase = is_pattern_match.group(1).strip()
                        # Split into words and add to protected text if they match object names
                        descriptive_words = descriptive_phrase.split()
                        protected_text.extend(descriptive_words)
                
                # Protect object names in description matching questions
                # Pattern: "matches this description: '{description}'" - protect description text
                description_match = re.search(r"matches this description:\s*'([^']+)'", question_with_colored_boxes)
                if description_match:
                    description_text = description_match.group(1).strip()
                    # Extract words from description that might be object names
                    description_words = re.findall(r'\b[a-z]+\b', description_text.lower())
                    protected_text.extend(description_words)
                
                # Replace object names in question text ONLY for spatial questions
                # For non-spatial questions, keep original question text unchanged (object names are part of the question description)
                if is_spatial:
                    # For spatial questions: replace object names with "object in {colored_box}"
                    for obj_name, colored_box in obj_to_colored_box.items():
                        # Skip replacement if the object name is part of protected descriptive text
                        if obj_name.lower() in [p.lower() for p in protected_text]:
                            continue
                        # Use regex with word boundaries to avoid partial matches
                        pattern = r'\b' + re.escape(obj_name) + r'\b'
                        replacement = f"object in {colored_box}"
                        question_with_colored_boxes = re.sub(pattern, replacement, question_with_colored_boxes)
                # For non-spatial questions: do NOT replace object names in question text
                # They are part of the question description (e.g., "designed as a container")
                
                # Replace object names in options suffix and convert to colored boxes
                # Skip adding options suffix for spatial questions (they don't need object choices)
                if not is_spatial:
                    if options_suffix:
                        # Extract object names from options suffix
                        options_text = options_suffix.split(":", 1)[1].strip() if ":" in options_suffix else ""
                        if options_text:
                            # Replace object names with colored boxes in options
                            for obj_name, colored_box in obj_to_colored_box.items():
                                pattern = r'\b' + re.escape(obj_name) + r'\b'
                                options_text = re.sub(pattern, colored_box, options_text)
                            options_suffix = "Objects to choose from: " + options_text
                        else:
                            # No options found, create from colored boxes
                            colored_box_options = ", ".join(obj_to_colored_box.values())
                            options_suffix = "Objects to choose from: " + colored_box_options
                    else:
                        # No options suffix found, append colored box choices
                        colored_box_options = ", ".join(obj_to_colored_box.values())
                        options_suffix = "Objects to choose from: " + colored_box_options
                    
                    # Combine question and options (only for non-spatial questions)
                    question_with_colored_boxes = f"{question_with_colored_boxes} {options_suffix}"
                else:
                    # For spatial questions, don't add options suffix - just use the question text
                    # Ensure question ends with proper punctuation
                    if not question_with_colored_boxes.rstrip().endswith('?'):
                        question_with_colored_boxes = question_with_colored_boxes.rstrip('.') + '?'
                
                # Keep original question/answer with object names (skip for spatial questions)
                if not is_spatial:
                    question['original_question'] = orig_question
                    question['original_answer'] = orig_answer
                
                # Replace with colored box versions for question/answer
                question['question'] = question_with_colored_boxes
                question['answer'] = answer_with_colored_boxes
                
                # Final validation: Check if answer appears in question text (shouldn't happen with proper protection)
                # This is a safety check - the root cause should be fixed above
                if not is_spatial:
                    # Extract question text only (exclude options)
                    question_text_only = question_with_colored_boxes.split("Objects to choose from:")[0].strip()
                    if answer_with_colored_boxes and answer_with_colored_boxes in question_text_only:
                        logger.warning(f"Malformed question detected (answer in question text): '{question_with_colored_boxes}' -> '{answer_with_colored_boxes}'. Skipping question.")
                        # Mark this question for removal
                        questions_to_remove.append(question)
                        continue
        
        # Remove malformed questions
        if questions_to_remove:
            logger.info(f"Removing {len(questions_to_remove)} malformed questions where answer appears in question")
            for q in questions_to_remove:
                all_questions.remove(q)
            # Re-index questions after removal
            for idx, question in enumerate(all_questions):
                question['question_index'] = idx
        
        # Save all questions
        questions_file = output_dir / "all_questions.json"
        with open(questions_file, 'w') as f:
            json.dump(all_questions, f, indent=2)
        
        # Save scene statistics
        stats_file = output_dir / "scene_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(all_scene_statistics, f, indent=2)
        
        # Calculate question category counts
        question_type_counts = {}
        for question in all_questions:
            q_category = question.get('question_category', 'unknown')
            question_type_counts[q_category] = question_type_counts.get(q_category, 0) + 1
        
        # Save metadata
        metadata = {
            "generation_info": {
                "total_questions": len(all_questions),
                "total_scenes": processed_count,
                "qa_space_data": qa_space_data,
                "questions_generated": len(all_questions),
                "scenes_with_questions": len(all_scene_statistics),
                "question_type_counts": question_type_counts,
                "random_seed": seed
            }
        }
        
        metadata_file = output_dir / "generation_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"QA generation complete: {len(all_questions)} questions generated for {processed_count} scenes")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate comprehensive QA with unified output')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Path to images directory')
    parser.add_argument('--output_dir', type=str, default='taxonomyQABench_realimage',
                       help='Output directory for QA files')
    parser.add_argument('--image_ids', type=str, nargs='*',
                       help='Specific image IDs to process (optional)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility (MUST be set before any random operations)
    import random
    random.seed(args.seed)
    logger.info(f"Random seed set to {args.seed} for reproducibility")
    
    # Also set numpy random seed if numpy is available
    try:
        import numpy as np
        np.random.seed(args.seed)
        logger.info(f"NumPy random seed set to {args.seed}")
    except ImportError:
        pass  # NumPy not available, skip
    
    # Create output directory in qa_gen (absolute path to avoid nested taxonomy_datagen issues)
    script_dir = Path(__file__).resolve().parent
    qa_gen_root = script_dir.parent.parent  # core -> scripts -> qa_gen
    # Always save in qa_gen folder - make paths relative to qa_gen_root
    if not Path(args.output_dir).is_absolute():
        output_dir = qa_gen_root / args.output_dir
    else:
        output_dir = Path(args.output_dir)
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = TaxonomyQABenchGenerator()
    
    # Process images
    images_path = Path(args.images_dir)
    generator.process_images_with_qa_space(
        images_path, output_dir, 
        image_ids=args.image_ids, max_files=args.max_files, seed=args.seed
    )


if __name__ == "__main__":
    main()
