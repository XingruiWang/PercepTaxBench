#!/usr/bin/env python3
"""
Create an open-answer variant of a taxonomy QA benchmark.

Transformations:
1. Remove "Option objects: ..." and "Objects to choose from: ..." from questions
2. Set answer = original_answer (object name)
3. Set choices = [] (no choices in prompt)

Usage:
    python create_open_answer_benchmark.py <source_benchmark_dir>

Examples:
    python create_open_answer_benchmark.py taxonomyQABench_realimage_final_polished
    python create_open_answer_benchmark.py taxonomyQABench_simimage_final
    python create_open_answer_benchmark.py taxonomyQABench_simimage_final_with_properties
    python create_open_answer_benchmark.py taxonomyQABench_realimage_final_polished_with_properties
"""

import argparse
import json
import re
import shutil
from pathlib import Path

# Question types that expect DIRECTIONAL answers (above/below, left/right, front/behind)
# Other spatial types (e.g. manual_spatial_relation, closer-to-camera) expect object names
SPATIAL_DIRECTIONAL_TYPES = {"spatial_above_below", "spatial_left_right", "spatial_front_behind"}
BOX_LABELS = {"Red box", "Green box", "Blue box", "Yellow box", "Orange box", "Pink box", "Purple box", "Magenta box"}


def _load_spatial_directional_types(bench_dir: Path) -> set:
    """Load spatial directional types from benchmark metadata; fallback to default."""
    for name in ("generation_metadata.json", "question_type_statistics.json"):
        path = bench_dir / name
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                types = data.get("spatial_directional_question_types")
                if types:
                    return set(types)
                # Infer from question_type_counts: spatial_above_below, spatial_left_right, spatial_front_behind
                counts = data.get("question_type_counts", {})
                if counts:
                    directional = {k for k in counts if k in SPATIAL_DIRECTIONAL_TYPES}
                    if directional:
                        return directional
            except (json.JSONDecodeError, KeyError):
                pass
    return SPATIAL_DIRECTIONAL_TYPES


def _resolve_answer(q: dict, spatial_directional_types: set) -> str:
    q_type = q.get("question_type", "")
    question = q.get("question", "").lower()
    answer = q.get("answer", "")
    original_answer = q.get("original_answer")
    target_object = q.get("target_object", "")
    box_to_object = q.get("box_to_object", {})

    if q_type in spatial_directional_types:
        dirs = {"above", "below", "left", "right", "front", "behind"}
        if answer and str(answer).lower() in dirs:
            return answer
        return original_answer or target_object or answer

    if "spatial" in q.get("question_category", "").lower() and "closer" in question:
        obj = target_object or original_answer or box_to_object.get(answer, answer)
        return obj if obj and obj not in BOX_LABELS else answer

    if answer in BOX_LABELS:
        return box_to_object.get(answer) or target_object or original_answer or answer
    return original_answer or target_object or answer


def process_questions(qs: list, spatial_directional_types: set) -> list:
    """Apply open-answer transformations to questions."""
    opt_pat = re.compile(r"\s*Option objects:\s*[^?\"]+?\??\s*", re.IGNORECASE)
    choose_pat = re.compile(r"\s*Objects to choose from:\s*[^\"]+", re.IGNORECASE)

    out = []
    for q in qs:
        q = dict(q)
        # Remove Option objects and Objects to choose from from question text
        t = q.get("question", "")
        t = opt_pat.sub("", t)
        t = choose_pat.sub("", t)
        t = t.rstrip(" ?")
        q["question"] = t
        q["answer"] = _resolve_answer(q, spatial_directional_types)
        # No choices in prompt
        q["choices"] = []
        out.append(q)
    return out


def main():
    parser = argparse.ArgumentParser(description="Create open-answer benchmark from source")
    parser.add_argument(
        "source",
        type=str,
        help="Source benchmark dir (e.g. taxonomyQABench_realimage_final_polished)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output dir (default: <source>_open_answer)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Process in place (do not copy; only transform all_questions.json)",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Overwrite output dir without prompting",
    )
    args = parser.parse_args()

    qa_gen = Path(__file__).resolve().parent
    source = qa_gen / args.source
    if not source.exists():
        parser.error(f"Source not found: {source}")

    if args.no_copy:
        out_dir = source
    else:
        out_dir = qa_gen / (args.output or f"{args.source}_open_answer")
        if out_dir.exists():
            if not args.yes:
                print(f"Output already exists: {out_dir}")
                overwrite = input("Overwrite? [y/N]: ").strip().lower()
                if overwrite != "y":
                    print("Aborted.")
                    return
            shutil.rmtree(out_dir)
        shutil.copytree(source, out_dir)
        print(f"Copied {source} -> {out_dir}")

    all_path = out_dir / "all_questions.json"
    if not all_path.exists():
        print(f"Warning: {all_path} not found, skipping.")
        return

    with open(all_path) as f:
        qs = json.load(f)

    spatial_directional_types = _load_spatial_directional_types(source)
    qs = process_questions(qs, spatial_directional_types)
    with open(all_path, "w") as f:
        json.dump(qs, f, indent=2)

    print(f"Processed {len(qs)} questions")

    meta_path = out_dir / "generation_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["note"] = (
            meta.get("note", "") + " | Open-answer variant: no choices, answer=original_answer"
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Updated {meta_path}")


if __name__ == "__main__":
    main()
