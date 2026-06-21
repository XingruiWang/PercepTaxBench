#!/usr/bin/env python3
"""
Calculate worst-case baseline accuracy for the QA benchmark.

Assumes uniform random guessing - probability of correct answer = 1 / num_choices
"""

import json
from pathlib import Path
from collections import Counter

def calculate_baseline_accuracy(questions_file):
    """Calculate baseline accuracy assuming uniform random guessing"""
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    total_questions = len(questions)
    accuracies = []
    choice_counts = Counter()
    
    for q in questions:
        choices = q.get('choices', [])
        num_choices = len(choices)
        
        if num_choices == 0:
            print(f"Warning: Question {q.get('question_index', 'unknown')} has no choices, skipping")
            continue
        
        choice_counts[num_choices] += 1
        accuracy = 1.0 / num_choices
        accuracies.append(accuracy)
    
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    
    print(f"\n{'='*60}")
    print(f"Baseline Accuracy Calculation (Random Guessing)")
    print(f"{'='*60}")
    print(f"Total questions analyzed: {len(accuracies)}")
    print(f"Average baseline accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"\nChoice distribution:")
    print(f"{'Num Choices':<15} {'Count':<15} {'Prob':<15} {'% of Total'}")
    print(f"{'-'*60}")
    
    for num_choices in sorted(choice_counts.keys()):
        count = choice_counts[num_choices]
        prob = 1.0 / num_choices
        pct = (count / len(accuracies)) * 100
        print(f"{num_choices:<15} {count:<15} {prob:.4f} ({prob*100:.2f}%) {pct:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"Summary: Random guessing would achieve ~{avg_accuracy*100:.2f}% accuracy")
    print(f"{'='*60}\n")
    
    return avg_accuracy, choice_counts

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate random guess baseline accuracy')
    parser.add_argument('--questions_file', 
                       type=Path,
                       default=Path("/path/to/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_v2_v3_filtered/all_questions.json"),
                       help='Path to questions JSON file')
    
    args = parser.parse_args()
    
    if not args.questions_file.exists():
        print(f"Error: Questions file not found: {args.questions_file}")
        exit(1)
    
    calculate_baseline_accuracy(args.questions_file)

