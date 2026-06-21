#!/usr/bin/env python3
"""
Regenerate taxonomyQABench_realimage_v3_unique_filtered from survey results.

Matches questions by (image_id, question_text) instead of question_index,
since question indices differ between survey and benchmark.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_question_text(question: str) -> str:
    """Normalize question text for matching (remove extra whitespace, etc.)"""
    if not question:
        return ""
    # Remove extra whitespace
    question = " ".join(question.split())
    return question.strip()


def create_filtered_benchmark(
    survey_file: Path,
    source_benchmark_dir: Path,
    output_dir: Path
):
    """
    Create filtered benchmark from survey results.
    
    Matches by (image_id, question_text) instead of question_index.
    """
    # Load survey results
    logger.info(f"Loading survey results from {survey_file}")
    with open(survey_file, 'r') as f:
        survey_data = json.load(f)
    
    assessments = survey_data.get('assessments', [])
    logger.info(f"Total assessments in survey: {len(assessments)}")
    
    # Get all High Quality - Correct questions
    high_quality = [
        a for a in assessments 
        if a.get('quality_assessment') == 'High Quality - Correct'
    ]
    logger.info(f"Found {len(high_quality)} 'High Quality - Correct' assessments")
    
    # Create a set of (image_id, normalized_question_text) for matching
    survey_matches = set()
    survey_entries = {}  # (image_id, normalized_question) -> assessment entry
    
    for assessment in high_quality:
        image_id = assessment.get('image_id', '')
        question = assessment.get('question', '')
        normalized_question = normalize_question_text(question)
        
        if image_id and normalized_question:
            key = (image_id, normalized_question)
            survey_matches.add(key)
            survey_entries[key] = assessment
    
    logger.info(f"Created {len(survey_matches)} unique (image_id, question) matches")
    
    # Load source benchmark questions
    source_questions_file = source_benchmark_dir / "all_questions.json"
    logger.info(f"Loading source benchmark from {source_questions_file}")
    with open(source_questions_file, 'r') as f:
        source_questions = json.load(f)
    
    logger.info(f"Total questions in source benchmark: {len(source_questions)}")
    
    # Match questions from source benchmark
    matched_questions = []
    matched_image_ids: Set[str] = set()
    
    for question in source_questions:
        image_id = question.get('image_id', '')
        question_text = question.get('question', '')
        normalized_question = normalize_question_text(question_text)
        
        if image_id and normalized_question:
            key = (image_id, normalized_question)
            if key in survey_matches:
                matched_questions.append(question)
                matched_image_ids.add(image_id)
    
    logger.info(f"Matched {len(matched_questions)} questions from source benchmark")
    logger.info(f"Unique images: {len(matched_image_ids)}")
    
    # Check for missing matches
    missing_matches = survey_matches - {
        (q.get('image_id', ''), normalize_question_text(q.get('question', '')))
        for q in matched_questions
    }
    
    if missing_matches:
        logger.warning(f"Found {len(missing_matches)} survey entries that couldn't be matched:")
        for image_id, question in list(missing_matches)[:5]:
            logger.warning(f"  image_id: {image_id}, question: {question[:80]}...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy matched questions
    output_questions_file = output_dir / "all_questions.json"
    with open(output_questions_file, 'w') as f:
        json.dump(matched_questions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(matched_questions)} questions to {output_questions_file}")
    
    # Copy images for matched questions
    source_images_dir = source_benchmark_dir / "images"
    copied_images = 0
    
    for image_id in matched_image_ids:
        source_image_dir = source_images_dir / image_id
        if source_image_dir.exists():
            dest_image_dir = output_images_dir / image_id
            if dest_image_dir.exists():
                shutil.rmtree(dest_image_dir)
            shutil.copytree(source_image_dir, dest_image_dir)
            copied_images += 1
        else:
            logger.warning(f"Image directory not found: {source_image_dir}")
    
    logger.info(f"Copied {copied_images} image directories")
    
    # Copy generation metadata if it exists
    source_metadata_file = source_benchmark_dir / "generation_metadata.json"
    if source_metadata_file.exists():
        dest_metadata_file = output_dir / "generation_metadata.json"
        shutil.copy2(source_metadata_file, dest_metadata_file)
        logger.info(f"Copied generation metadata")
    
    # Create summary
    summary = {
        "source_benchmark": str(source_benchmark_dir),
        "survey_file": str(survey_file),
        "total_high_quality_in_survey": len(high_quality),
        "total_matched_questions": len(matched_questions),
        "total_unique_images": len(matched_image_ids),
        "total_images_copied": copied_images,
        "matching_method": "image_id + normalized_question_text"
    }
    
    summary_file = output_dir / "filtering_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved filtering summary to {summary_file}")
    
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    print(f"High Quality - Correct in survey: {len(high_quality)}")
    print(f"Matched questions: {len(matched_questions)}")
    print(f"Unique images: {len(matched_image_ids)}")
    print(f"Images copied: {copied_images}")
    if missing_matches:
        print(f"WARNING: {len(missing_matches)} survey entries could not be matched")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create filtered benchmark from survey results"
    )
    parser.add_argument(
        "--survey_file",
        type=Path,
        default=Path("survey_results_v3_unique/taxonomy_qa_survey_Jonathan.json"),
        help="Path to survey results JSON file"
    )
    parser.add_argument(
        "--source_benchmark",
        type=Path,
        default=Path("taxonomyQABench_realimage_v3_unique"),
        help="Path to source benchmark directory"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("taxonomyQABench_realimage_v3_unique_filtered"),
        help="Path to output filtered benchmark directory"
    )
    
    args = parser.parse_args()
    
    create_filtered_benchmark(
        survey_file=args.survey_file,
        source_benchmark_dir=args.source_benchmark,
        output_dir=args.output_dir
    )

