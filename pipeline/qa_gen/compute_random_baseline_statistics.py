#!/usr/bin/env python3

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def categorize_question(question: Dict) -> str:
    """
    Categorize a question into one of four categories:
    - taxonomy_reasoning
    - taxonomy_description (excluding description_matching)
    - spatial_relation
    - object_description (only description_matching)
    """
    question_category = question.get("question_category", "")
    question_type = question.get("question_type", "")
    
    if question_type == "description_matching":
        return "object_description"
    elif question_category == "taxonomy_reasoning":
        return "taxonomy_reasoning"
    elif question_category == "taxonomy_description":
        return "taxonomy_description"
    elif question_category == "spatial_relation":
        return "spatial_relation"
    else:
        return "unknown"


def count_choices(question: Dict) -> int:
    """
    Count the number of choices for a question.
    For spatial relation questions, always returns 2 (spatial terms like above/below, left/right).
    For other questions, counts colored boxes from question text or choices field.
    """
    question_category = question.get("question_category", "")
    question_type = question.get("question_type", "")
    
    # For spatial relation questions, always 2 choices (spatial terms)
    if question_category == "spatial_relation" and not question_type.startswith("manual_"):
        return 2
    
    # For other questions, check if choices are colored boxes
    choices = question.get("choices", [])
    if not choices:
        return 0
    
    # Check if choices are colored boxes (e.g., "Red box", "Green box", "Blue box")
    colored_box_pattern = r"(Red|Green|Blue|Yellow|Orange|Purple|Pink|Brown|Black|White)\s+box"
    colored_boxes = [c for c in choices if isinstance(c, str) and re.match(colored_box_pattern, c, re.IGNORECASE)]
    
    if colored_boxes:
        return len(colored_boxes)
    
    # If not colored boxes, check question text for colored boxes
    question_text = question.get("question", "")
    if "Objects to choose from:" in question_text or "choose from:" in question_text.lower():
        # Extract colored boxes from question text
        matches = re.findall(colored_box_pattern, question_text, re.IGNORECASE)
        if matches:
            return len(set(matches))  # Count unique colored boxes
    
    # Fallback: return length of choices field
    return len(choices)


def compute_statistics(questions: List[Dict], benchmark_name: str) -> Dict[str, Any]:
    """
    Compute random chance baseline statistics for a benchmark.
    Excludes manual questions (question_type starting with "manual_").
    """
    # Filter out manual questions
    non_manual_questions = [
        q for q in questions 
        if not q.get("question_type", "").startswith("manual_")
    ]
    
    stats = {
        "benchmark": benchmark_name,
        "total_questions": len(questions),
        "total_questions_excluding_manual": len(non_manual_questions),
        "categories": {}
    }
    
    category_data = defaultdict(lambda: {
        "questions": [],
        "question_types": defaultdict(list),
        "total_choices": 0,
        "question_count": 0
    })
    
    overall_total_choices = 0
    overall_question_count = len(non_manual_questions)
    
    for question in non_manual_questions:
        category = categorize_question(question)
        num_choices = count_choices(question)
        question_type = question.get("question_type", "unknown")
        
        overall_total_choices += num_choices
        
        category_data[category]["questions"].append(question)
        category_data[category]["question_types"][question_type].append(question)
        category_data[category]["total_choices"] += num_choices
        category_data[category]["question_count"] += 1
    
    for category, data in category_data.items():
        if data["question_count"] == 0:
            continue
            
        avg_choices = data["total_choices"] / data["question_count"] if data["question_count"] > 0 else 0
        random_baseline = 1.0 / avg_choices if avg_choices > 0 else 0
        
        category_stats = {
            "total_questions": data["question_count"],
            "average_choices": round(avg_choices, 4),
            "random_chance_baseline": round(random_baseline, 6),
            "question_types": {}
        }
        
        for question_type, type_questions in data["question_types"].items():
            # Skip manual question types
            if question_type.startswith("manual_"):
                continue
                
            type_choices_sum = sum(count_choices(q) for q in type_questions)
            type_avg_choices = type_choices_sum / len(type_questions) if type_questions else 0
            type_baseline = 1.0 / type_avg_choices if type_avg_choices > 0 else 0
            
            category_stats["question_types"][question_type] = {
                "count": len(type_questions),
                "average_choices": round(type_avg_choices, 4),
                "random_chance_baseline": round(type_baseline, 6)
            }
        
        stats["categories"][category] = category_stats
    
    overall_avg_choices = overall_total_choices / overall_question_count if overall_question_count > 0 else 0
    overall_baseline = 1.0 / overall_avg_choices if overall_avg_choices > 0 else 0
    
    stats["overall"] = {
        "total_questions": overall_question_count,
        "average_choices": round(overall_avg_choices, 4),
        "random_chance_baseline": round(overall_baseline, 6)
    }
    
    return stats


def print_summary(all_stats: Dict[str, Any]):
    """Print a human-readable summary of the statistics."""
    print("=" * 80)
    print("RANDOM CHANCE BASELINE STATISTICS")
    print("=" * 80)
    print()
    
    for benchmark_name, stats in all_stats.items():
        print(f"Benchmark: {benchmark_name}")
        print(f"Total Questions: {stats['total_questions']}")
        print(f"Total Questions (excluding manual): {stats.get('total_questions_excluding_manual', stats['total_questions'])}")
        print()
        
        if "overall" in stats:
            print(f"  OVERALL STATISTICS (All Questions)")
            print(f"    Total Questions: {stats['overall']['total_questions']}")
            print(f"    Average Choices: {stats['overall']['average_choices']}")
            print(f"    Random Chance Baseline: {stats['overall']['random_chance_baseline']:.4%}")
            print()
        
        for category, cat_stats in stats["categories"].items():
            print(f"  Category: {category.upper()}")
            print(f"    Total Questions: {cat_stats['total_questions']}")
            print(f"    Average Choices: {cat_stats['average_choices']}")
            print(f"    Random Chance Baseline: {cat_stats['random_chance_baseline']:.4%}")
            print()
            
            if cat_stats["question_types"]:
                print(f"    Question Types Breakdown:")
                for qtype, type_stats in sorted(cat_stats["question_types"].items()):
                    print(f"      {qtype}:")
                    print(f"        Count: {type_stats['count']}")
                    print(f"        Avg Choices: {type_stats['average_choices']}")
                    print(f"        Baseline: {type_stats['random_chance_baseline']:.4%}")
                print()
        
        print("-" * 80)
        print()


def main():
    script_dir = Path(__file__).resolve().parent
    
    benchmark_paths = {
        "sim_image": script_dir / "taxonomyQABench_simimage_final" / "all_questions.json",
        "real_image": script_dir / "taxonomyQABench_realimage_final_polished" / "all_questions.json"
    }
    
    all_stats = {}
    
    for benchmark_name, benchmark_path in benchmark_paths.items():
        if not benchmark_path.exists():
            print(f"Warning: Benchmark file not found: {benchmark_path}")
            continue
        
        print(f"Loading {benchmark_name} benchmark...")
        with open(benchmark_path, 'r') as f:
            questions = json.load(f)
        
        print(f"Computing statistics for {benchmark_name}...")
        stats = compute_statistics(questions, benchmark_name)
        all_stats[benchmark_name] = stats
    
    output_file = script_dir / "random_chance_baseline_statistics.json"
    print(f"Saving statistics to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print()
    print_summary(all_stats)
    print(f"Detailed statistics saved to: {output_file}")


if __name__ == "__main__":
    main()

