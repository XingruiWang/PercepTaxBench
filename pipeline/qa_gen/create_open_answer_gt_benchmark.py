#!/usr/bin/env python3
"""
Create open_answer_gt benchmark: open_answer style (object-name answers, no choices)
+ ground truth properties from with_properties (appended to question, box colors only).

Output: taxonomyQABench_*_open_answer_gt
- Answer: object name (or direction for spatial)
- Choices: []
- Question: base question + property block (affordance/material/function/physical by box color)
- Properties use Red box, Green box, etc. - no object names revealed
"""

import argparse
import json
import re
import shutil
from pathlib import Path


PROPERTY_PREFIXES = (
    "affordance property:",
    "function property:",
    "material property:",
    "physical property:",
)


def _extract_property_block(question_with_props: str) -> str:
    """Extract property block (box-keyed properties) from with_properties question."""
    block = []
    for line in question_with_props.split("\n"):
        s = line.strip()
        if not s:
            continue
        if any(s.lower().startswith(p) for p in PROPERTY_PREFIXES):
            block.append(s)
    return "\n".join(block) if block else ""


def create_open_answer_gt(
    open_answer_dir: Path,
    with_properties_dir: Path,
    output_dir: Path,
) -> None:
    """Merge open_answer + with_properties into open_answer_gt."""
    oa_path = open_answer_dir / "all_questions.json"
    wp_path = with_properties_dir / "all_questions.json"

    if not oa_path.exists() or not wp_path.exists():
        raise FileNotFoundError(f"Missing all_questions.json in one of the dirs")

    with open(oa_path) as f:
        oa_questions = json.load(f)
    with open(wp_path) as f:
        wp_questions = json.load(f)

    # Match by question_index
    wp_by_idx = {q["question_index"]: q for q in wp_questions}

    out_questions = []
    for q in oa_questions:
        idx = q["question_index"]
        q_out = dict(q)
        base_question = q.get("question", "").strip()

        wp = wp_by_idx.get(idx)
        if wp:
            prop_block = _extract_property_block(wp.get("question", ""))
            if prop_block:
                q_out["question"] = base_question + "\n" + prop_block

        q_out["choices"] = []
        out_questions.append(q_out)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "all_questions.json", "w") as f:
        json.dump(out_questions, f, indent=2)

    # Copy other files from open_answer (images, metadata, etc.)
    for name in ["images", "generation_metadata.json", "scene_statistics.json", "question_type_statistics.json"]:
        src = open_answer_dir / name
        if src.exists():
            dst = output_dir / name
            if src.is_dir():
                if not dst.exists():
                    shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    # Copy all_questions_reasoning.json if present
    rpath = open_answer_dir / "all_questions_reasoning.json"
    if rpath.exists():
        shutil.copy2(rpath, output_dir / "all_questions_reasoning.json")

    # Update metadata note
    meta_path = output_dir / "generation_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["note"] = meta.get("note", "") + " | open_answer_gt: object-name answers + ground truth properties (box-keyed)"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Created {output_dir} with {len(out_questions)} questions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Create real image open_answer_gt")
    parser.add_argument("--sim", action="store_true", help="Create sim image open_answer_gt")
    parser.add_argument("--both", action="store_true", help="Create both real and sim")
    args = parser.parse_args()

    qa_gen = Path(__file__).resolve().parent

    if args.both or (not args.real and not args.sim):
        args.real = True
        args.sim = True

    if args.real:
        create_open_answer_gt(
            qa_gen / "taxonomyQABench_realimage_final_polished_open_answer",
            qa_gen / "taxonomyQABench_realimage_final_polished_with_properties",
            qa_gen / "taxonomyQABench_realimage_final_polished_open_answer_gt",
        )

    if args.sim:
        create_open_answer_gt(
            qa_gen / "taxonomyQABench_simimage_final_open_answer",
            qa_gen / "taxonomyQABench_simimage_final_with_properties",
            qa_gen / "taxonomyQABench_simimage_final_open_answer_gt",
        )


if __name__ == "__main__":
    main()
