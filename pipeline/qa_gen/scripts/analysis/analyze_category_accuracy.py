#!/usr/bin/env python3
"""
Analyze VLM evaluation results by question category.

This script:
1. Loads VLM evaluation results (with index, correct, answers, predict)
2. Loads question metadata (with question_index, question_category)
3. Matches questions by index
4. Calculates accuracy per question category
5. Saves detailed results to JSON
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import sys


def load_json(file_path: Path) -> Any:
    """Load JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def analyze_category_accuracy(
    eval_results_path: Path,
    questions_path: Path,
    output_path: Path = None
) -> Dict[str, Any]:
    """
    Analyze accuracy by question category.
    
    Args:
        eval_results_path: Path to VLM eval results JSON
        questions_path: Path to all_questions.json
        output_path: Optional path to save results
        
    Returns:
        Dictionary with category-wise accuracy statistics
    """
    print(f"Loading evaluation results from: {eval_results_path}")
    eval_data = load_json(eval_results_path)
    
    print(f"Loading questions from: {questions_path}")
    questions_data = load_json(questions_path)
    
    # Create index -> question mapping
    questions_by_index = {}
    for q in questions_data:
        idx = q.get('question_index')
        if idx is not None:
            questions_by_index[idx] = q
    
    print(f"Found {len(questions_by_index)} questions with indices")
    
    # Extract eval results
    eval_answers = eval_data.get('answers', [])
    total_eval = eval_data.get('total', len(eval_answers))
    correct_eval = eval_data.get('correct', sum(1 for a in eval_answers if a.get('correct')))
    
    print(f"Evaluation results: {correct_eval}/{total_eval} correct ({100*correct_eval/total_eval:.2f}%)")
    
    # Initialize category statistics
    category_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'accuracy': 0.0,
        'questions': []
    })
    
    # Also track detailed question types if available
    detailed_type_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'accuracy': 0.0
    })
    
    # Process each evaluation result
    matched_count = 0
    unmatched_count = 0
    
    for eval_result in eval_answers:
        idx = eval_result.get('index')
        if idx is None:
            continue
            
        question = questions_by_index.get(idx)
        if not question:
            unmatched_count += 1
            continue
        
        matched_count += 1
        category = question.get('question_category', 'unknown')
        detailed_type = question.get('question_type', 'unknown')
        is_correct = eval_result.get('correct', False)
        
        # Update category stats
        category_stats[category]['total'] += 1
        if is_correct:
            category_stats[category]['correct'] += 1
        else:
            category_stats[category]['incorrect'] += 1
        
        # Update detailed type stats
        detailed_type_stats[detailed_type]['total'] += 1
        if is_correct:
            detailed_type_stats[detailed_type]['correct'] += 1
        else:
            detailed_type_stats[detailed_type]['incorrect'] += 1
        
        # Store question details
        category_stats[category]['questions'].append({
            'index': idx,
            'correct': is_correct,
            'ground_truth': eval_result.get('answers', ''),
            'prediction': eval_result.get('predict', ''),
            'question': question.get('question', '')[:100],  # Truncate for readability
            'question_type': detailed_type
        })
    
    # Calculate accuracies
    for category in category_stats:
        stats = category_stats[category]
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total']
    
    for detailed_type in detailed_type_stats:
        stats = detailed_type_stats[detailed_type]
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total']
    
    # Prepare results
    results = {
        'summary': {
            'total_evaluated': total_eval,
            'total_correct': correct_eval,
            'overall_accuracy': correct_eval / total_eval if total_eval > 0 else 0.0,
            'matched_questions': matched_count,
            'unmatched_questions': unmatched_count
        },
        'category_accuracy': {
            category: {
                'total': stats['total'],
                'correct': stats['correct'],
                'incorrect': stats['incorrect'],
                'accuracy': stats['accuracy'],
                'percentage': f"{stats['accuracy']*100:.2f}%"
            }
            for category, stats in sorted(category_stats.items())
        },
        'detailed_type_accuracy': {
            qtype: {
                'total': stats['total'],
                'correct': stats['correct'],
                'incorrect': stats['incorrect'],
                'accuracy': stats['accuracy'],
                'percentage': f"{stats['accuracy']*100:.2f}%"
            }
            for qtype, stats in sorted(detailed_type_stats.items()) if stats['total'] > 0
        }
    }
    
    # Add detailed questions (optional - can be large)
    # Uncomment if you want detailed question-level breakdown
    # results['detailed_questions_by_category'] = {
    #     category: stats['questions']
    #     for category, stats in category_stats.items()
    # }
    
    # Print summary
    print("\n" + "="*80)
    print("CATEGORY-WISE ACCURACY SUMMARY")
    print("="*80)
    print(f"\nOverall: {correct_eval}/{total_eval} ({results['summary']['overall_accuracy']*100:.2f}%)")
    print(f"Matched: {matched_count}, Unmatched: {unmatched_count}\n")
    
    print(f"{'Category':<20} {'Total':<8} {'Correct':<8} {'Incorrect':<10} {'Accuracy':<10}")
    print("-" * 80)
    
    # Sort by total count (descending)
    sorted_categories = sorted(
        category_stats.items(),
        key=lambda x: x[1]['total'],
        reverse=True
    )
    
    for category, stats in sorted_categories:
        print(f"{category:<20} {stats['total']:<8} {stats['correct']:<8} {stats['incorrect']:<10} {stats['accuracy']*100:>6.2f}%")
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze VLM evaluation results by question category',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real image benchmark
  python scripts/analysis/analyze_category_accuracy.py \\
      --eval-results /path/to/VLMEvalKit/outputs/GeminiPro2-5/GeminiPro2-5_TaxonomyBench_score.json \\
      --questions taxonomyQABench_realimage_v2/all_questions.json \\
      --output scripts/analysis/category_accuracy_realimage.json
  
  # Sim image benchmark
  python scripts/analysis/analyze_category_accuracy.py \\
      --eval-results /path/to/VLMEvalKit/outputs/Model_TaxonomyBench_score.json \\
      --questions taxonomyQABench_simimage/all_questions.json \\
      --output scripts/analysis/category_accuracy_simimage.json
  
  # With custom output path
  python scripts/analysis/analyze_category_accuracy.py \\
      --eval-results eval.json \\
      --questions questions.json \\
      --output results/category_accuracy.json
        """
    )
    
    parser.add_argument(
        '--eval-results',
        type=Path,
        required=True,
        help='Path to VLM evaluation results JSON file'
    )
    
    parser.add_argument(
        '--questions',
        type=Path,
        required=True,
        help='Path to all_questions.json file with question metadata'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for results JSON (default: category_accuracy_<timestamp>.json in same dir as eval-results)'
    )
    
    parser.add_argument(
        '--metadata',
        type=Path,
        default=None,
        help='Optional: Path to generation_metadata.json (currently unused, for future enhancements)'
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        timestamp = Path(args.eval_results).stat().st_mtime
        output_dir = args.eval_results.parent
        output_name = f"category_accuracy_{Path(args.eval_results).stem}.json"
        args.output = output_dir / output_name
    
    results = analyze_category_accuracy(
        args.eval_results,
        args.questions,
        args.output
    )
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

