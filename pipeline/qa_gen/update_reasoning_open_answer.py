#!/usr/bin/env python3
"""
Update all_questions_reasoning.json in an open_answer benchmark so that
final_answer and reasoning_text use object names (or directions for spatial)
instead of box labels (Red box, Green box, etc.).

Uses all_questions.json as source of truth for correct answers.
"""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bench_dir",
        type=str,
        default="taxonomyQABench_simimage_final_open_answer",
        nargs="?",
        help="Open-answer benchmark directory",
    )
    args = parser.parse_args()

    qa_gen = Path(__file__).resolve().parent
    bench = qa_gen / args.bench_dir

    questions_path = bench / "all_questions.json"
    reasoning_path = bench / "all_questions_reasoning.json"

    if not questions_path.exists():
        print(f"Missing: {questions_path}")
        return
    if not reasoning_path.exists():
        print(f"Missing: {reasoning_path}")
        return

    with open(questions_path) as f:
        questions = json.load(f)

    # question_index -> correct answer (object name or direction)
    idx_to_answer = {q["question_index"]: q["answer"] for q in questions}

    with open(reasoning_path) as f:
        reasoning_list = json.load(f)

    updated = 0
    for i, entry in enumerate(reasoning_list):
        idx = entry.get("question_index", i)
        correct_answer = idx_to_answer.get(idx)
        if correct_answer is None:
            continue

        for phase in entry.get("reasoning", []):
            if phase.get("phase") == "answer":
                old_answer = phase.get("final_answer", "")
                if old_answer != correct_answer:
                    phase["final_answer"] = correct_answer
                    updated += 1

                    # Update reasoning_text: replace <answer> old with <answer> new
                    rt = entry.get("reasoning_text", "")
                    if rt and old_answer != correct_answer:
                        # Replace in answer section: <answer> old — or <answer> old\n
                        pattern = r"(<answer>\s*)" + re.escape(old_answer) + r"(\s*[—\-]|\s*\n)"
                        new_rt = re.sub(pattern, r"\1" + correct_answer + r"\2", rt, count=1)
                        if new_rt != rt:
                            entry["reasoning_text"] = new_rt
                break

    with open(reasoning_path, "w") as f:
        json.dump(reasoning_list, f, indent=2)

    print(f"Updated {updated} reasoning entries in {reasoning_path}")


if __name__ == "__main__":
    main()
