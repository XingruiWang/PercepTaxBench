#!/usr/bin/env python3
"""
Fix Bbox Color Mismatch Script

This script fixes existing questions where colored boxes don't match the bbox image colors.
It ensures:
1. Choices order matches detection order (bbox image color order)
2. Color assignments match the RGB order: red, green, blue, yellow, magenta, cyan...

Usage:
    python fix_bbox_color_mismatch.py --questions_path <path> --annotations_dir <path>
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict

# RGB color order matching visualization_utils.py
COLOR_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "lime", "light blue", "purple", "pink", "turquoise"]


def load_detection_order(image_id: str, annotations_dir: Path) -> List[str]:
    """Load detection order from annotation files (order determines bbox colors)"""
    image_dir = annotations_dir / image_id
    
    # Look for annotation files
    annotation_files = []
    annotations_path = image_dir / "annotations"
    if annotations_path.exists():
        refined_files = list(annotations_path.glob("*_refined.json"))
        if refined_files:
            annotation_files.extend(refined_files)
        else:
            json_files = list(annotations_path.glob("*.json"))
            annotation_files.extend([f for f in json_files if not f.name.endswith("_refined.json")])
    
    if not annotation_files:
        json_files = list(image_dir.glob("*.json"))
        annotation_files.extend(json_files)
    
    if not annotation_files:
        return []
    
    # Load first annotation file
    with open(annotation_files[0], 'r') as f:
        annotation_data = json.load(f)
    
    # Extract detection order
    detections = annotation_data.get('detections', annotation_data.get('detected_objects', []))
    detection_order = []
    seen = set()
    
    for detection in detections:
        if isinstance(detection, dict):
            class_name = detection.get('class_name', detection.get('class', ''))
            if class_name and class_name != 'unknown' and class_name not in seen:
                detection_order.append(class_name)
                seen.add(class_name)
        elif isinstance(detection, str) and detection not in seen:
            detection_order.append(detection)
            seen.add(detection)
    
    return detection_order


def get_colored_box_name(obj_idx: int) -> str:
    """Get colored box name based on index (matching visualization_utils.py RGB order)"""
    color_name = COLOR_NAMES[obj_idx % len(COLOR_NAMES)]
    return f"{color_name.capitalize()} box"


def fix_question_colors(question: Dict, detection_order: List[str]) -> Tuple[bool, Dict]:
    """Fix colored box assignments in a question to match detection order"""
    choices = question.get('choices', question.get('objects', []))
    if not choices:
        return False, question
    
    # Create mapping from detection order to colors
    # Reorder choices to match detection order
    ordered_choices = []
    for obj in detection_order:
        if obj in choices:
            ordered_choices.append(obj)
    
    # Add any remaining choices that weren't in detection order
    for obj in choices:
        if obj not in ordered_choices:
            ordered_choices.append(obj)
    
    # If order didn't change, skip
    if ordered_choices == choices:
        return False, question
    
    # Recalculate colored box assignments
    obj_to_colored_box = {obj: get_colored_box_name(i) for i, obj in enumerate(ordered_choices)}
    
    # Update answer
    target_object = question.get('target_object', question.get('answer_object', ''))
    if target_object in obj_to_colored_box:
        new_answer = obj_to_colored_box[target_object]
    else:
        # Try to find current answer in mapping
        current_answer = question.get('answer', '')
        # Extract object name from "Red box" format
        current_box_color = current_answer.replace(' box', '').lower()
        try:
            current_idx = COLOR_NAMES.index(current_box_color)
            if current_idx < len(ordered_choices):
                new_answer = get_colored_box_name(current_idx)
            else:
                new_answer = current_answer  # Keep if can't determine
        except ValueError:
            new_answer = current_answer  # Keep if can't parse
    
    # Update question text - replace colored box options
    question_text = question.get('question', '')
    if 'Option objects:' in question_text:
        # Replace the options list
        parts = question_text.split('Option objects:')
        if len(parts) == 2:
            colored_boxes = [obj_to_colored_box[obj] for obj in ordered_choices]
            options_str = ", ".join(colored_boxes)
            question_text = parts[0].strip() + f" Option objects: {options_str}"
    
    # Create updated question
    updated_question = question.copy()
    updated_question['choices'] = ordered_choices
    updated_question['objects'] = ordered_choices  # Also update objects to match
    updated_question['question'] = question_text
    updated_question['answer'] = new_answer
    
    return True, updated_question


def main():
    parser = argparse.ArgumentParser(description="Fix bbox color mismatches in questions")
    parser.add_argument(
        "--questions_path",
        type=str,
        default="/path/to/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage/all_questions.json",
        help="Path to all_questions.json"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="/path/to/SpatialReasonerDataGen/qa_gen/openimages_unified_output",
        help="Path to annotations directory"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run - show what would be fixed without saving"
    )
    
    args = parser.parse_args()
    
    questions_path = Path(args.questions_path)
    annotations_dir = Path(args.annotations_dir)
    
    # Load questions
    print(f"Loading questions from {questions_path}")
    with open(questions_path, 'r') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Group questions by image_id
    questions_by_image = {}
    for q in questions:
        image_id = q.get('image_id', '')
        if image_id:
            if image_id not in questions_by_image:
                questions_by_image[image_id] = []
            questions_by_image[image_id].append(q)
    
    print(f"Found questions for {len(questions_by_image)} images")
    
    # Fix each image's questions
    fixed_count = 0
    skipped_images = []
    
    for image_id, image_questions in questions_by_image.items():
        # Load detection order
        detection_order = load_detection_order(image_id, annotations_dir)
        
        if not detection_order:
            skipped_images.append(image_id)
            continue
        
        # Fix each question for this image
        for q_idx, question in enumerate(image_questions):
            changed, fixed_question = fix_question_colors(question, detection_order)
            if changed:
                questions[questions.index(question)] = fixed_question
                fixed_count += 1
                print(f"Fixed question {question.get('question_index', q_idx)} for image {image_id}")
                print(f"  Old choices: {question.get('choices', [])}")
                print(f"  New choices: {fixed_question.get('choices', [])}")
                print(f"  Old answer: {question.get('answer', '')}")
                print(f"  New answer: {fixed_question.get('answer', '')}")
    
    if skipped_images:
        print(f"\nSkipped {len(skipped_images)} images (no annotation files found)")
    
    print(f"\nFixed {fixed_count} questions")
    
    if not args.dry_run and fixed_count > 0:
        # Backup original
        backup_path = questions_path.with_suffix('.json.backup')
        print(f"Creating backup: {backup_path}")
        questions_path.rename(backup_path)
        
        # Save fixed questions
        print(f"Saving fixed questions to {questions_path}")
        with open(questions_path, 'w') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print("Done!")
    elif args.dry_run:
        print("\nDry run complete. Use without --dry_run to apply changes.")


if __name__ == "__main__":
    main()

