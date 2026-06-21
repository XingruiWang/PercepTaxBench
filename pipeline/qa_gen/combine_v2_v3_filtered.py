#!/usr/bin/env python3
"""
Combine taxonomyQABench_realimage_v2_filtered and taxonomyQABench_realimage_v3_unique_filtered
into a single benchmark, removing duplicates based on (question_text, choices, image_id).
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_choices(choices: List) -> Tuple:
    """Normalize choices list for comparison (sort and convert to tuple)."""
    if not choices:
        return tuple()
    return tuple(sorted(str(c).lower().strip() for c in choices if c))


def normalize_question_text(question_text: str) -> str:
    """
    Normalize question text for comparison.
    Handles variations like "Option objects:" vs "Objects to choose from:".
    """
    if not question_text:
        return ""
    
    text = question_text.strip().lower()
    
    # Normalize "Option objects:" and "Objects to choose from:" to a common format
    # Replace both with a placeholder, then remove it (they're equivalent)
    text = text.replace("option objects:", "objects to choose from:")
    
    # Also handle case variations
    text = text.replace("option object:", "objects to choose from:")
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text


def create_question_key(question: Dict) -> Tuple:
    """
    Create a unique key for deduplication based on:
    - question text (normalized, handling "Option objects:" vs "Objects to choose from:")
    - choices (normalized and sorted)
    - image_id
    """
    question_text = normalize_question_text(question.get('question', ''))
    choices = normalize_choices(question.get('choices', []))
    image_id = question.get('image_id', '').strip()
    
    return (question_text, choices, image_id)


def combine_benchmarks(
    v2_dir: Path,
    v3_dir: Path,
    output_dir: Path
):
    """
    Combine two filtered benchmarks, removing duplicates.
    """
    # Load v2 questions
    v2_questions_file = v2_dir / "all_questions.json"
    logger.info(f"Loading v2 questions from {v2_questions_file}")
    with open(v2_questions_file, 'r') as f:
        v2_questions = json.load(f)
    logger.info(f"Loaded {len(v2_questions)} questions from v2")
    
    # Load v3 questions
    v3_questions_file = v3_dir / "all_questions.json"
    logger.info(f"Loading v3 questions from {v3_questions_file}")
    with open(v3_questions_file, 'r') as f:
        v3_questions = json.load(f)
    logger.info(f"Loaded {len(v3_questions)} questions from v3")
    
    # Track unique questions by key
    unique_questions = {}  # key -> question
    question_keys_seen: Set[Tuple] = set()
    v2_count = 0
    v3_count = 0
    v2_duplicates = 0
    v3_duplicates = 0
    
    # Process v2 questions first
    for question in v2_questions:
        key = create_question_key(question)
        if key not in question_keys_seen:
            unique_questions[key] = question
            question_keys_seen.add(key)
            v2_count += 1
        else:
            v2_duplicates += 1
    
    logger.info(f"V2: {v2_count} unique, {v2_duplicates} duplicates")
    
    # Process v3 questions (v3 takes precedence if duplicate)
    for question in v3_questions:
        key = create_question_key(question)
        if key not in question_keys_seen:
            unique_questions[key] = question
            question_keys_seen.add(key)
            v3_count += 1
        else:
            # Replace with v3 version if duplicate (v3 takes precedence)
            unique_questions[key] = question
            v3_duplicates += 1
    
    logger.info(f"V3: {v3_count} unique, {v3_duplicates} duplicates")
    
    # Convert to list and reassign question indices sequentially
    combined_questions = list(unique_questions.values())
    logger.info(f"Total unique questions: {len(combined_questions)}")
    
    # Reassign question indices sequentially starting from 0
    for idx, question in enumerate(combined_questions):
        question['question_index'] = idx
    
    logger.info(f"Reassigned question indices from 0 to {len(combined_questions) - 1}")
    
    # Collect unique image IDs
    unique_image_ids: Set[str] = set()
    for question in combined_questions:
        image_id = question.get('image_id', '').strip()
        if image_id:
            unique_image_ids.add(image_id)
    
    logger.info(f"Unique images: {len(unique_image_ids)}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined questions
    output_questions_file = output_dir / "all_questions.json"
    with open(output_questions_file, 'w') as f:
        json.dump(combined_questions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(combined_questions)} questions to {output_questions_file}")
    
    # Copy images from both sources (v3 takes precedence if same image_id)
    copied_images = 0
    v2_images_dir = v2_dir / "images"
    v3_images_dir = v3_dir / "images"
    
    # First, copy v2 images
    for image_id in unique_image_ids:
        v2_image_dir = v2_images_dir / image_id
        v3_image_dir = v3_images_dir / image_id
        dest_image_dir = output_images_dir / image_id
        
        # Prefer v3 if exists, otherwise use v2
        if v3_image_dir.exists():
            if dest_image_dir.exists():
                shutil.rmtree(dest_image_dir)
            shutil.copytree(v3_image_dir, dest_image_dir)
            copied_images += 1
        elif v2_image_dir.exists():
            if dest_image_dir.exists():
                shutil.rmtree(dest_image_dir)
            shutil.copytree(v2_image_dir, dest_image_dir)
            copied_images += 1
        else:
            logger.warning(f"Image directory not found in either source: {image_id}")
    
    logger.info(f"Copied {copied_images} image directories")
    
    # Copy metadata files (prefer v3 if exists, otherwise v2)
    metadata_files = ["generation_metadata.json", "question_type_statistics.json", "scene_statistics.json"]
    for metadata_file in metadata_files:
        v3_file = v3_dir / metadata_file
        v2_file = v2_dir / metadata_file
        dest_file = output_dir / metadata_file
        
        if v3_file.exists():
            shutil.copy2(v3_file, dest_file)
            logger.info(f"Copied {metadata_file} from v3")
        elif v2_file.exists():
            shutil.copy2(v2_file, dest_file)
            logger.info(f"Copied {metadata_file} from v2")
    
    # Create summary
    summary = {
        "source_benchmarks": {
            "v2": str(v2_dir),
            "v3": str(v3_dir)
        },
        "v2_questions": len(v2_questions),
        "v3_questions": len(v3_questions),
        "v2_unique_added": v2_count,
        "v3_unique_added": v3_count,
        "v2_duplicates_skipped": v2_duplicates,
        "v3_duplicates_replaced": v3_duplicates,
        "total_unique_questions": len(combined_questions),
        "total_unique_images": len(unique_image_ids),
        "total_images_copied": copied_images,
        "deduplication_key": "question_text + normalized_choices + image_id"
    }
    
    summary_file = output_dir / "combining_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved combining summary to {summary_file}")
    
    print("\n" + "="*60)
    print("COMBINING SUMMARY")
    print("="*60)
    print(f"V2 questions: {len(v2_questions)}")
    print(f"V3 questions: {len(v3_questions)}")
    print(f"V2 unique added: {v2_count}")
    print(f"V3 unique added: {v3_count}")
    print(f"V2 duplicates skipped: {v2_duplicates}")
    print(f"V3 duplicates replaced: {v3_duplicates}")
    print(f"Total unique questions: {len(combined_questions)}")
    print(f"Unique images: {len(unique_image_ids)}")
    print(f"Images copied: {copied_images}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Combine two filtered benchmarks"
    )
    parser.add_argument(
        "--v2_dir",
        type=Path,
        default=Path("taxonomyQABench_realimage_v2_filtered"),
        help="Path to v2 filtered benchmark directory"
    )
    parser.add_argument(
        "--v3_dir",
        type=Path,
        default=Path("taxonomyQABench_realimage_v3_unique_filtered"),
        help="Path to v3 filtered benchmark directory"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("taxonomyQABench_realimage_v2_v3_filtered"),
        help="Path to output combined benchmark directory"
    )
    
    args = parser.parse_args()
    
    combine_benchmarks(
        v2_dir=args.v2_dir,
        v3_dir=args.v3_dir,
        output_dir=args.output_dir
    )

