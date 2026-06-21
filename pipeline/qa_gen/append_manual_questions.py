#!/usr/bin/env python3
"""
Append manual questions from manual_qa_PK_complete to taxonomyQABench_realimage_manual.
- Use original_answer as the correct answer choice
- Sync images from taxonomyQABench_realimage_final_polished/images
"""

import json
import shutil
from pathlib import Path

QA_GEN = Path(__file__).resolve().parent
MANUAL_SOURCE = QA_GEN / "additional_qa_results" / "manual_qa_PK_complete_20251112_000232.json"
MANUAL_BENCH = QA_GEN / "taxonomyQABench_realimage_manual"
FINAL_POLISHED_IMAGES = QA_GEN / "taxonomyQABench_realimage_final_polished" / "images"

# Map source question_category to taxonomyQABench style
CATEGORY_MAP = {
    "function": "taxonomy_reasoning",
    "spatial_relation": "spatial_relation",
    "material": "taxonomy_description",
    "description": "taxonomy_description",
    "affordance": "taxonomy_reasoning",
    "counterfactual": "taxonomy_reasoning",
    "other": "taxonomy_reasoning",
    "taxonomy_reasoning": "taxonomy_reasoning",
}

TYPE_MAP = {
    "function": "manual_taxonomy_reasoning",
    "spatial_relation": "manual_spatial_relation",
    "material": "manual_taxonomy_description",
    "description": "manual_taxonomy_description",
    "affordance": "manual_taxonomy_reasoning",
    "counterfactual": "manual_taxonomy_reasoning",
    "other": "manual_taxonomy_reasoning",
    "taxonomy_reasoning": "manual_taxonomy_reasoning",
}


def main():
    with open(MANUAL_SOURCE, "r") as f:
        data = json.load(f)

    questions = data["questions"]
    image_ids = set()

    out_questions = []
    for i, q in enumerate(questions):
        # Use original_answer as the correct answer choice
        q_out = {
            "question": q.get("question", q.get("original_question", "")),
            "answer": q.get("original_answer", q.get("answer", "")),
            "original_question": q.get("original_question", q.get("question", "")),
            "original_answer": q.get("original_answer", q.get("answer", "")),
            "answer_object": q.get("original_answer", q.get("answer", "")),
            "target_object": q.get("original_answer", q.get("answer", "")),
            "objects": q.get("objects", []),
            "choices": q.get("choices", []),
            "reasoning": q.get("reasoning", ""),
            "question_category": CATEGORY_MAP.get(
                q.get("question_category", ""), "taxonomy_reasoning"
            ),
            "question_type": TYPE_MAP.get(
                q.get("question_category", ""), "manual_taxonomy_reasoning"
            ),
            "image_id": q.get("image_id", ""),
            "image_path": f"{q.get('image_id', '')}/bbox.jpg",
            "box_to_object": q.get("box_to_object", {}),
            "created_timestamp": q.get("created_timestamp", ""),
            "created_by": q.get("created_by", "PK"),
            "is_manual": True,
            "manual_entry": True,
            "question_index": i,
            "rephrased_with_template": False,
        }
        out_questions.append(q_out)
        if q.get("image_id"):
            image_ids.add(q["image_id"])

    # Write all_questions.json
    all_path = MANUAL_BENCH / "all_questions.json"
    with open(all_path, "w") as f:
        json.dump(out_questions, f, indent=2)

    print(f"Wrote {len(out_questions)} questions to {all_path}")

    # Sync images from final_polished to manual/images
    manual_images = MANUAL_BENCH / "images"
    manual_images.mkdir(parents=True, exist_ok=True)

    synced = 0
    for img_id in image_ids:
        src_dir = FINAL_POLISHED_IMAGES / img_id
        dst_dir = manual_images / img_id
        if not src_dir.exists():
            print(f"Warning: source image dir not found: {src_dir}")
            continue
        if dst_dir.exists():
            # Already present; optionally refresh
            continue
        try:
            shutil.copytree(src_dir, dst_dir)
            synced += 1
        except OSError as e:
            print(f"Warning: could not copy {src_dir} -> {dst_dir}: {e}")

    print(f"Synced {synced} new image directories from {FINAL_POLISHED_IMAGES}")

    # Update generation_metadata.json
    cat_counts = {}
    type_counts = {}
    for q in out_questions:
        cat = q["question_category"]
        typ = q["question_type"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        type_counts[typ] = type_counts.get(typ, 0) + 1

    meta = {
        "generation_info": {
            "total_questions": len(out_questions),
            "total_scenes": len(image_ids),
            "sources": {
                "manual_extracted_from": "manual_qa_PK_complete_20251112_000232",
                "images_from": "taxonomyQABench_realimage_final_polished",
            },
        },
        "question_type_counts": type_counts,
        "question_category_counts": cat_counts,
        "note": "Manual QA pairs from PK complete; images from final_polished; original_answer as correct.",
    }

    meta_path = MANUAL_BENCH / "generation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Updated {meta_path}")


if __name__ == "__main__":
    main()
