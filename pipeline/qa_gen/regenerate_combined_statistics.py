#!/usr/bin/env python3
"""
Regenerate statistics files for the combined benchmark based on actual data.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

def regenerate_statistics(benchmark_dir: Path):
    """Regenerate all statistics files from the combined benchmark."""
    
    # Load questions
    questions_file = benchmark_dir / "all_questions.json"
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Count unique QA pairs
    qa_pairs = set()
    for q in questions:
        question_text = q.get('question', '').strip()
        answer = q.get('answer', '').strip()
        image_id = q.get('image_id', '').strip()
        if question_text and answer and image_id:
            qa_pairs.add((question_text.lower(), answer.lower(), image_id))
    
    print(f"Unique (question, answer, image_id) pairs: {len(qa_pairs)}")
    
    # Remap legacy categories (e.g., merge capability into function)
    legacy_category_remap = {
        "capability": "function",
    }

    # Map simplified categories to new taxonomy groupings
    new_category_map = {
        "affordance": "taxonomy_description",
        "material": "taxonomy_description",
        "function": "taxonomy_description",
        "description": "taxonomy_description",
        "taxonomy_description": "taxonomy_description",
        "repurposing": "taxonomy_reasoning",
        "latent": "taxonomy_reasoning",
        "counterfactual": "taxonomy_reasoning",
        "compositional": "taxonomy_reasoning",
        "capability": "taxonomy_reasoning",
        "taxonomy_reasoning": "taxonomy_reasoning",
        "spatial": "spatial_relation",
        "spatial_relation": "spatial_relation",
    }

    # Update questions in-place with new categories and count stats
    question_category_counts = Counter()
    for q in questions:
        legacy_category = q.get('question_category', 'unknown')
        simplified_category = legacy_category_remap.get(legacy_category, legacy_category)
        new_category = new_category_map.get(simplified_category, simplified_category)
        q['question_category'] = new_category
        question_category_counts[new_category] += 1
    
    # Count unique images/scenes
    unique_images = set()
    image_to_questions = defaultdict(int)
    for q in questions:
        image_id = q.get('image_id', '').strip()
        if image_id:
            unique_images.add(image_id)
            image_to_questions[image_id] += 1
    
    print(f"Unique images: {len(unique_images)}")
    
    # Generate question_type_statistics.json
    # Note: v2_v3_filtered uses question_category, not question_type
    question_type_stats = {
        "total_questions": len(questions),
        "unique_qa_pairs": len(qa_pairs),
        "question_category_counts": dict(sorted(question_category_counts.items(), key=lambda x: x[1], reverse=True)),
        "unique_question_categories": len(question_category_counts),
        "unique_images": len(unique_images),
        "note": "Statistics regenerated from combined v2_filtered and v3_unique_filtered benchmarks"
    }
    
    stats_file = benchmark_dir / "question_type_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(question_type_stats, f, indent=2)
    print(f"Saved question_type_statistics.json")

    # Persist updated question categories back to all_questions.json
    with open(questions_file, 'w') as f:
        json.dump(questions, f, indent=2)
    print("Updated all_questions.json with recalculated question categories")
    
    # Generate scene_statistics.json
    scene_stats = []
    for image_id in sorted(unique_images):
        scene_stats.append({
            "scene_id": image_id,
            "image_id": image_id,
            "question_count": image_to_questions[image_id]
        })
    
    scene_stats_file = benchmark_dir / "scene_statistics.json"
    with open(scene_stats_file, 'w') as f:
        json.dump(scene_stats, f, indent=2)
    print(f"Saved scene_statistics.json with {len(scene_stats)} scenes")
    
    # Update generation_metadata.json (keep existing structure but update totals)
    metadata_file = benchmark_dir / "generation_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Update with correct totals
    if "generation_info" not in metadata:
        metadata["generation_info"] = {}
    
    metadata["generation_info"]["total_questions"] = len(questions)
    metadata["generation_info"]["unique_qa_pairs"] = len(qa_pairs)
    metadata["generation_info"]["total_scenes"] = len(unique_images)
    metadata["generation_info"]["question_category_counts"] = dict(sorted(question_category_counts.items(), key=lambda x: x[1], reverse=True))
    metadata["generation_info"]["unique_question_categories"] = len(question_category_counts)
    metadata["generation_info"]["regenerated"] = True
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated generation_metadata.json")
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICS SUMMARY")
    print("="*60)
    print(f"Total questions: {len(questions)}")
    print(f"Unique QA pairs: {len(qa_pairs)}")
    print(f"Unique images: {len(unique_images)}")
    print(f"\nQuestion categories:")
    for cat, count in sorted(question_category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Regenerate statistics for combined benchmark")
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        default=Path("taxonomyQABench_realimage_v2_v3_filtered"),
        help="Path to benchmark directory"
    )
    
    args = parser.parse_args()
    regenerate_statistics(args.benchmark_dir)

