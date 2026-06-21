#!/usr/bin/env python3
"""
Build the taxonomyQABench_realimage_final benchmark by combining the
v2_v3_filtered benchmark with additional high-quality questions that
were manually validated in survey_results_new against the v2 dataset.

Steps performed:
1. Load the curated base benchmark (v2_v3_filtered).
2. Collect all survey assessments marked "High Quality - Correct".
3. Pull the corresponding questions from taxonomyQABench_realimage_v2.
4. Normalize legacy metadata (question categories, missing fields).
5. Deduplicate against the base benchmark (question text + choices).
6. Reindex questions and write the combined dataset.
7. Copy required image assets and regenerate lightweight statistics.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Directory layout
QA_GEN_ROOT = Path(__file__).resolve().parent
BASE_BENCH_DIR = QA_GEN_ROOT / "taxonomyQABench_realimage_v2_v3_filtered"
SOURCE_BENCH_DIR = QA_GEN_ROOT / "taxonomyQABench_realimage_v2"
SURVEY_RESULTS_DIR = QA_GEN_ROOT / "survey_results_new"
OUTPUT_BENCH_DIR = QA_GEN_ROOT / "taxonomyQABench_realimage_final"
MANUAL_QA_PATH = QA_GEN_ROOT / "additional_qa_results" / "manual_qa_PK.json"
MANUAL_QUESTION_LIMIT = None

# Ensure modules path for category utilities
import sys

sys.path.append(str(QA_GEN_ROOT / "scripts" / "modules" / "qa_modules"))
from question_type_grouping import get_simplified_question_type  # type: ignore
from legacy_question_type_utils import infer_legacy_question_type  # type: ignore


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

NEW_CATEGORY_CHOICES = [
    "taxonomy_description",
    "taxonomy_reasoning",
    "spatial_relation",
    "other",
]

ALLOWED_CATEGORIES = {
    "taxonomy_description",
    "taxonomy_reasoning",
    "spatial_relation",
}

# Map legacy/manual labels to the new taxonomy groupings
LEGACY_CATEGORY_MAP = {
    "affordance": "taxonomy_description",
    "description": "taxonomy_description",
    "material": "taxonomy_description",
    "repurposing": "taxonomy_reasoning",
    "compositional": "taxonomy_reasoning",
    "function": "taxonomy_reasoning",
    "counterfactual": "taxonomy_reasoning",
    "capability": "taxonomy_reasoning",
    "latent": "taxonomy_reasoning",
    "spatial": "spatial_relation",
    "spatial_relation": "spatial_relation",
    "causal and spatial": "spatial_relation",
    "": None,
    None: None,
}


def normalize_category(question: Dict) -> Optional[str]:
    """Map legacy question categories to the consolidated taxonomy labels."""
    qtype = (question.get("question_type") or "").strip()
    if qtype:
        mapped = get_simplified_question_type(qtype)
        if mapped:
            return mapped

    raw_category = question.get("question_category")
    normalized = LEGACY_CATEGORY_MAP.get(raw_category, raw_category)
    if normalized in NEW_CATEGORY_CHOICES:
        return normalized
    if normalized in (None, ""):
        return None
    return "other"


def fill_missing_question_type(question: Dict) -> None:
    """Ensure question_type is populated for legacy entries."""
    if question.get("manual_entry") or question.get("is_manual"):
        return

    question_type = question.get("question_type")
    if not question_type:
        inferred = infer_legacy_question_type(question)
        if inferred:
            question_type = inferred
            question["question_type"] = inferred
        else:
            normalized_category = question.get("question_category")
            if normalized_category not in ALLOWED_CATEGORIES:
                normalized_category = normalize_category(question) or "taxonomy_description"
            question["question_category"] = normalized_category
            legacy_type = {
                "taxonomy_description": "legacy_taxonomy_description",
                "taxonomy_reasoning": "legacy_taxonomy_reasoning",
                "spatial_relation": "legacy_spatial_relation",
            }[normalized_category]
            question["question_type"] = legacy_type
            question["legacy_entry"] = True
            return

    # Ensure the category aligns with the inferred question_type.
    category = determine_category_from_type(question_type, question.get("question_category"))
    question["question_category"] = category


def determine_category_from_type(question_type: str, existing_category: Optional[str]) -> str:
    """
    Map a detailed question_type to one of the consolidated categories.
    """
    if question_type:
        simplified = get_simplified_question_type(question_type)
        if simplified in ALLOWED_CATEGORIES:
            return simplified

    if existing_category in ALLOWED_CATEGORIES:
        return existing_category  # Already compliant.

    if question_type.startswith("spatial_") or question_type == "legacy_spatial_relation":
        return "spatial_relation"

    if question_type.startswith(("repurposing_", "counterfactual_", "latent_", "functional_", "compositional_",
                                 "taxonomy_reasoning", "legacy_taxonomy_reasoning")):
        return "taxonomy_reasoning"

    return "taxonomy_description"


def question_key(question: Dict) -> Tuple[str, Tuple[str, ...], Tuple[str, ...], Optional[str]]:
    question_text = (question.get("question") or "").strip()
    objects = tuple(question.get("objects", []) or [])
    choices = tuple(question.get("choices", []) or [])
    image_id = extract_image_id(question)
    return question_text, objects, choices, image_id


def gather_high_quality_indices(survey_dir: Path) -> Set[int]:
    """Collect question indices marked High Quality - Correct across all surveys."""
    indices: Set[int] = set()
    for survey_file in survey_dir.glob("**/*.json"):
        try:
            data = json.loads(survey_file.read_text())
        except Exception as exc:
            print(f"Warning: failed to read {survey_file}: {exc}")
            continue

        for assessment in data.get("assessments", []):
            if assessment.get("quality_assessment") != "High Quality - Correct":
                continue
            idx = assessment.get("question_index")
            if isinstance(idx, int):
                indices.add(idx)
            else:
                try:
                    indices.add(int(idx))
                except Exception:
                    print(f"Warning: invalid question_index '{idx}' in {survey_file}")
    return indices


def ensure_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_base_images(base_dir: Path, output_dir: Path) -> None:
    src_images = base_dir / "images"
    dst_images = output_dir / "images"
    shutil.copytree(src_images, dst_images, dirs_exist_ok=True)


def extract_image_id(question: Dict) -> Optional[str]:
    image_id = question.get("image_id")
    if image_id:
        return image_id
    image_path = question.get("image_path")
    if image_path:
        return Path(image_path).parts[0]
    return None


def copy_missing_images(
    image_ids: Iterable[str],
    base_images_dir: Path,
    source_images_dir: Path,
    output_images_dir: Path,
) -> None:
    for image_id in image_ids:
        dst_dir = output_images_dir / image_id
        if dst_dir.exists():
            continue
        src_dir = source_images_dir / image_id
        if not src_dir.exists():
            # Try legacy location if needed
            print(f"Warning: missing source image directory for {image_id}")
            continue
        shutil.copytree(src_dir, dst_dir)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_manual_questions(limit: int = MANUAL_QUESTION_LIMIT) -> List[Dict]:
    """Load manually authored questions to be appended to the benchmark."""
    if not MANUAL_QA_PATH.exists():
        print(f"  Warning: manual QA file not found at {MANUAL_QA_PATH}")
        return []

    payload = json.loads(MANUAL_QA_PATH.read_text())
    manual_questions = payload.get("questions", [])
    if limit is not None and len(manual_questions) > limit:
        manual_questions = manual_questions[:limit]

    loaded: List[Dict] = []
    for entry in manual_questions:
        question = deepcopy(entry)
        question["is_manual"] = True
        question["manual_entry"] = True

        # Ensure consolidated category mapping
        category = normalize_category(question)
        if category not in ALLOWED_CATEGORIES:
            category = LEGACY_CATEGORY_MAP.get(category, category)
        if category not in ALLOWED_CATEGORIES:
            category = "taxonomy_description"
        question["question_category"] = category

        question["question_type"] = f"manual_{category.replace(' ', '_')}"

        question.setdefault("box_to_object", question.get("box_to_object") or {})
        question.setdefault("objects", question.get("objects") or [])
        question.setdefault("choices", question.get("choices") or [])

        loaded.append(question)

    print(f"  Loaded {len(loaded)} manual questions from {MANUAL_QA_PATH.name}.")
    return loaded


DESCRIPTION_PROMPT_REGEX = re.compile(
    r"(Which object matches this description:\s*['\"])([^'\"]+)(['\"])",
    re.IGNORECASE,
)


def _clean_redundant_made_of(text: str) -> str:
    if not text:
        return text
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered.startswith("made of "):
        remainder = stripped[8:].lstrip()
        remainder_lower = remainder.lower()
        word_count = len(remainder.split())
        if word_count > 2 or any(
            marker in remainder_lower
            for marker in (
                " made ",
                " are ",
                " is ",
                " was ",
                " were ",
                " usually ",
                " typically ",
                " generally ",
            )
        ):
            return remainder
    return stripped


def sanitize_description_question(question: Dict) -> None:
    if not isinstance(question, dict):
        return
    if question.get("question_type") != "description_matching" and (
        question.get("question_category") != "taxonomy_description"
        or "matches this description" not in (question.get("question") or "")
    ):
        # Skip non-description questions
        return

    text = question.get("question")
    if isinstance(text, str):
        def _replace(match: re.Match) -> str:
            prefix, description, suffix = match.groups()
            cleaned = _clean_redundant_made_of(description)
            return f"{prefix}{cleaned}{suffix}"

        new_text = DESCRIPTION_PROMPT_REGEX.sub(_replace, text, count=1)
        question["question"] = new_text

    metadata = question.get("question_metadata")
    if isinstance(metadata, dict) and metadata.get("description"):
        metadata["description"] = _clean_redundant_made_of(metadata["description"])


# ------------------------------------------------------------------------------
# Main build routine
# ------------------------------------------------------------------------------

def main() -> None:
    print("🚀 Building taxonomyQABench_realimage_final ...")

    # Load base benchmark (already curated and filtered)
    base_questions = json.loads((BASE_BENCH_DIR / "all_questions.json").read_text())
    print(f"  Loaded base benchmark with {len(base_questions)} questions.")

    # Gather high quality question indices from manual surveys
    high_quality_indices = gather_high_quality_indices(SURVEY_RESULTS_DIR)
    print(f"  Found {len(high_quality_indices)} high quality survey assessments.")

    # Load full v2 question set to pull additional questions
    source_questions = json.loads((SOURCE_BENCH_DIR / "all_questions.json").read_text())
    question_by_index = {q.get("question_index"): q for q in source_questions}
    print(f"  Loaded source benchmark with {len(source_questions)} questions.")

    combined_questions: List[Dict] = deepcopy(base_questions)
    existing_keys = {question_key(q) for q in combined_questions}

    duplicates_skipped = 0
    missing_indices: List[int] = []
    survey_added_questions: List[Dict] = []

    for idx in sorted(high_quality_indices):
        source_question = question_by_index.get(idx)
        if source_question is None:
            missing_indices.append(idx)
            continue

        question_copy = deepcopy(source_question)

        # Normalize metadata for compatibility
        category = normalize_category(question_copy)
        if category is None:
            # Skip if we cannot map it reliably
            print(
                f"  Skipping question_index {idx} - unable to normalize category "
                f"(original: {question_copy.get('question_category')})"
            )
            duplicates_skipped += 1
            continue
        question_copy["question_category"] = category
        if question_copy.get("question_type") is None:
            question_copy["question_type"] = ""

        key = question_key(question_copy)
        if key in existing_keys:
            duplicates_skipped += 1
            continue

        combined_questions.append(question_copy)
        existing_keys.add(key)
        survey_added_questions.append(question_copy)

    print(f"  Added {len(survey_added_questions)} new questions from surveys.")
    print(f"  Skipped {duplicates_skipped} duplicates and/or unmappable entries.")
    if missing_indices:
        print(f"  Warning: {len(missing_indices)} indices missing from v2 dataset.")

    manual_questions = load_manual_questions()
    manual_added = len(manual_questions)
    if manual_added:
        combined_questions.extend(manual_questions)
        print(f"  Added {manual_added} manual questions to the combined benchmark.")

    # Deduplicate across the entire combined benchmark (question + choices)
    dedup_seen: Set[Tuple[str, Tuple[str, ...], Optional[str]]] = set()
    deduped_questions: List[Dict] = []
    duplicates_removed_post = 0
    for question in combined_questions:
        key = (
            (question.get("question") or "").strip(),
            tuple(question.get("choices", []) or []),
            extract_image_id(question),
        )
        if key in dedup_seen:
            duplicates_removed_post += 1
            continue
        dedup_seen.add(key)
        deduped_questions.append(question)

    if duplicates_removed_post:
        print(f"  Removed {duplicates_removed_post} duplicate questions post-merge.")

    added_questions_combined = survey_added_questions + manual_questions
    combined_questions = deduped_questions

    # Populate missing question_type metadata for legacy questions.
    for question in combined_questions:
        fill_missing_question_type(question)
        sanitize_description_question(question)

    # Prepare output directory and copy assets
    ensure_output_dir(OUTPUT_BENCH_DIR)
    copy_base_images(BASE_BENCH_DIR, OUTPUT_BENCH_DIR)

    # Copy any new images needed
    base_image_ids = {extract_image_id(q) for q in base_questions}
    new_image_ids = {
        img_id
        for img_id in (extract_image_id(q) for q in added_questions_combined)
        if img_id and img_id not in base_image_ids
    }
    if new_image_ids:
        print(f"  Copying {len(new_image_ids)} additional image directories.")
        copy_missing_images(
            new_image_ids,
            BASE_BENCH_DIR / "images",
            SOURCE_BENCH_DIR / "images",
            OUTPUT_BENCH_DIR / "images",
        )

    # Reindex questions
    for new_index, question in enumerate(combined_questions):
        question["question_index"] = new_index

    # Write combined all_questions.json
    write_json(OUTPUT_BENCH_DIR / "all_questions.json", combined_questions)

    # Generate statistics files
    category_counts = Counter(
        q.get("question_category", "other") for q in combined_questions
    )
    unique_images = {
        img_id for img_id in (extract_image_id(q) for q in combined_questions) if img_id
    }
    question_stats = {
        "total_questions": len(combined_questions),
        "unique_qa_pairs": len(combined_questions),
        "question_category_counts": dict(sorted(category_counts.items())),
        "unique_question_categories": len(category_counts),
        "unique_images": len(unique_images),
        "note": "Combined v2_v3_filtered with high-quality survey additions and manual QA entries",
    }
    write_json(OUTPUT_BENCH_DIR / "question_type_statistics.json", question_stats)

    scene_counts = Counter(
        img_id for img_id in (extract_image_id(q) for q in combined_questions) if img_id
    )
    scene_stats = [
        {"scene_id": scene_id, "image_id": scene_id, "question_count": count}
        for scene_id, count in sorted(scene_counts.items())
    ]
    write_json(OUTPUT_BENCH_DIR / "scene_statistics.json", scene_stats)

    combining_summary = {
        "source_benchmarks": {
            "base": "taxonomyQABench_realimage_v2_v3_filtered",
            "survey_source": "taxonomyQABench_realimage_v2 + survey_results_new",
            "manual_source": MANUAL_QA_PATH.name,
        },
        "base_questions": len(base_questions),
        "high_quality_candidates": len(high_quality_indices),
        "high_quality_added": len(survey_added_questions),
        "manual_added": manual_added,
        "duplicates_skipped": duplicates_skipped,
        "duplicates_removed_postprocess": duplicates_removed_post,
        "missing_indices": missing_indices,
        "final_total_questions": len(combined_questions),
        "final_unique_images": len(unique_images),
    }
    write_json(OUTPUT_BENCH_DIR / "combining_summary.json", combining_summary)

    generation_metadata = {
        "generation_info": {
            "total_questions": len(combined_questions),
            "total_scenes": len(unique_images),
            "sources": {
                "base_benchmark": str(BASE_BENCH_DIR.name),
                "additional_questions_from": str(SOURCE_BENCH_DIR.name),
                "survey_directory": str(SURVEY_RESULTS_DIR.relative_to(QA_GEN_ROOT)),
            },
        },
        "notes": [
            f"Base benchmark size: {len(base_questions)} questions.",
            f"High quality additions: {len(survey_added_questions)}.",
            f"Manual additions: {manual_added}.",
            f"Duplicates skipped: {duplicates_skipped}.",
        ],
    }
    write_json(OUTPUT_BENCH_DIR / "generation_metadata.json", generation_metadata)

    print(
        f"✅ Finished building taxonomyQABench_realimage_final with "
        f"{len(combined_questions)} questions across {len(unique_images)} images."
    )


if __name__ == "__main__":
    main()

