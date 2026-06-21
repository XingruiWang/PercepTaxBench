#!/usr/bin/env python3
"""
Create taxonomyQABench_realimage_v2_filtered from survey results.

Collects all "High Quality - Correct" assessments from all JSON files in survey_results_new,
matches by question_index to taxonomyQABench_realimage_v2, and creates a filtered benchmark.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_all_survey_files(survey_dir: Path) -> List[Path]:
    """Find all JSON survey files recursively in survey_dir."""
    survey_files = []
    for json_file in survey_dir.rglob("*.json"):
        # Skip auto_saved files if they're duplicates
        if "auto_saved" in str(json_file):
            continue
        survey_files.append(json_file)
    return survey_files


def collect_high_quality_assessments(survey_files: List[Path]) -> Dict[int, Dict]:
    """
    Collect all "High Quality - Correct" assessments from all survey files.
    
    Returns a dictionary mapping question_index -> assessment entry.
    If the same question_index appears multiple times, keeps the first one found.
    """
    high_quality_by_index = {}
    total_assessments = 0
    high_quality_count = 0
    
    for survey_file in survey_files:
        logger.info(f"Processing survey file: {survey_file}")
        try:
            with open(survey_file, 'r') as f:
                survey_data = json.load(f)
            
            assessments = survey_data.get('assessments', [])
            total_assessments += len(assessments)
            
            for assessment in assessments:
                if assessment.get('quality_assessment') == 'High Quality - Correct':
                    high_quality_count += 1
                    question_index = assessment.get('question_index')
                    
                    if question_index is not None:
                        # Only keep first occurrence if duplicate
                        if question_index not in high_quality_by_index:
                            high_quality_by_index[question_index] = assessment
        except Exception as e:
            logger.error(f"Error processing {survey_file}: {e}")
    
    logger.info(f"Total assessments processed: {total_assessments}")
    logger.info(f"Total 'High Quality - Correct' assessments: {high_quality_count}")
    logger.info(f"Unique question indices: {len(high_quality_by_index)}")
    
    return high_quality_by_index


def create_filtered_benchmark(
    survey_dir: Path,
    source_benchmark_dir: Path,
    output_dir: Path
):
    """
    Create filtered benchmark from survey results.
    
    Matches by question_index to questions in v2.
    """
    # Find all survey files
    survey_files = find_all_survey_files(survey_dir)
    logger.info(f"Found {len(survey_files)} survey files")
    
    # Collect all high quality assessments
    high_quality_by_index = collect_high_quality_assessments(survey_files)
    
    # Load source benchmark questions
    source_questions_file = source_benchmark_dir / "all_questions.json"
    logger.info(f"Loading source benchmark from {source_questions_file}")
    with open(source_questions_file, 'r') as f:
        source_questions = json.load(f)
    
    logger.info(f"Total questions in source benchmark: {len(source_questions)}")
    
    # Match questions from source benchmark by question_index
    matched_questions = []
    matched_image_ids: Set[str] = set()
    matched_indices: Set[int] = set()
    
    for question in source_questions:
        question_index = question.get('question_index')
        
        if question_index is not None and question_index in high_quality_by_index:
            matched_questions.append(question)
            matched_image_ids.add(question.get('image_id', ''))
            matched_indices.add(question_index)
    
    logger.info(f"Matched {len(matched_questions)} questions from source benchmark")
    logger.info(f"Unique images: {len(matched_image_ids)}")
    
    # Check for missing matches
    missing_indices = set(high_quality_by_index.keys()) - matched_indices
    if missing_indices:
        logger.warning(f"Found {len(missing_indices)} survey question indices that couldn't be matched:")
        for idx in list(missing_indices)[:10]:
            assessment = high_quality_by_index[idx]
            logger.warning(f"  question_index: {idx}, image_id: {assessment.get('image_id')}, question: {assessment.get('question', '')[:80]}...")
    
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
        if not image_id:
            continue
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
    
    # Copy other metadata files if they exist
    for metadata_file in ["question_type_statistics.json", "scene_statistics.json"]:
        source_file = source_benchmark_dir / metadata_file
        if source_file.exists():
            dest_file = output_dir / metadata_file
            shutil.copy2(source_file, dest_file)
            logger.info(f"Copied {metadata_file}")
    
    # Create summary
    summary = {
        "source_benchmark": str(source_benchmark_dir),
        "survey_dir": str(survey_dir),
        "survey_files_processed": len(survey_files),
        "total_high_quality_assessments": len(high_quality_by_index),
        "total_matched_questions": len(matched_questions),
        "total_unique_images": len(matched_image_ids),
        "total_images_copied": copied_images,
        "matching_method": "question_index",
        "missing_matches": len(missing_indices)
    }
    
    summary_file = output_dir / "filtering_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved filtering summary to {summary_file}")
    
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    print(f"Survey files processed: {len(survey_files)}")
    print(f"High Quality - Correct assessments: {len(high_quality_by_index)}")
    print(f"Matched questions: {len(matched_questions)}")
    print(f"Unique images: {len(matched_image_ids)}")
    print(f"Images copied: {copied_images}")
    if missing_indices:
        print(f"WARNING: {len(missing_indices)} survey question indices could not be matched")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create filtered benchmark from survey results"
    )
    parser.add_argument(
        "--survey_dir",
        type=Path,
        default=Path("survey_results_new"),
        help="Path to survey results directory"
    )
    parser.add_argument(
        "--source_benchmark",
        type=Path,
        default=Path("taxonomyQABench_realimage_v2"),
        help="Path to source benchmark directory"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("taxonomyQABench_realimage_v2_filtered"),
        help="Path to output filtered benchmark directory"
    )
    
    args = parser.parse_args()
    
    create_filtered_benchmark(
        survey_dir=args.survey_dir,
        source_benchmark_dir=args.source_benchmark,
        output_dir=args.output_dir
    )

