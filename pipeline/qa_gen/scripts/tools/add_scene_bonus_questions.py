#!/usr/bin/env python3
"""
Add one additional question per scene to an existing taxonomyQABench_simimage dataset.

For each scene (grouped by `source_scene_id`) we clone a randomly selected question,
lightly adjust the question text so it remains unique, and append it to the dataset.
The script recomputes question indices and optionally updates generation metadata.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append one additional random question per scene to sim image QA dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the existing all_questions.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to overwriting --input.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional generation_metadata.json path to update alongside the questions file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def invert_box_mapping(box_to_object: Optional[Dict[str, str]]) -> Dict[str, List[str]]:
    inverted: Dict[str, List[str]] = defaultdict(list)
    if isinstance(box_to_object, dict):
        for label, obj in box_to_object.items():
            if obj:
                inverted[obj].append(label)
    return inverted


def format_choice_labels(choices: Iterable[str], box_to_object: Optional[Dict[str, str]]) -> List[str]:
    inverted = invert_box_mapping(box_to_object)
    labels: List[str] = []
    for obj in choices or []:
        candidate_labels = inverted.get(obj)
        labels.append(candidate_labels[0] if candidate_labels else obj)
    return labels


def sanitise_question_text(question: str) -> str:
    if not question:
        return question
    return " ".join(question.split())


def augment_question_text(question: str, choices: List[str]) -> str:
    if not isinstance(question, str) or not question.strip():
        return question

    stripped = question.strip()

    options_prefix = "Objects to choose from:"
    suffix = ""
    main = stripped
    if options_prefix in stripped:
        main, _, suffix = stripped.partition(options_prefix)
        main = main.strip()
        suffix = suffix.strip()

    if main.lower().startswith("in this scene"):
        augmented_main = sanitise_question_text(main)
    else:
        if main:
            first_char = main[0]
            remainder = main[1:] if len(main) > 1 else ""
            lowered = (first_char.lower() + remainder) if first_char.isupper() else main
        else:
            lowered = main
        augmented_main = sanitise_question_text(f"In this scene, {lowered}")

    if choices:
        formatted_choices = ", ".join(choices)
        full = f"{augmented_main} {options_prefix} {formatted_choices}".strip()
    else:
        full = augmented_main

    return sanitise_question_text(full)


def augment_original_question(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return text
    stripped = text.strip()
    if stripped.lower().startswith("in this scene"):
        return sanitise_question_text(stripped)
    first_char = stripped[0]
    remainder = stripped[1:] if len(stripped) > 1 else ""
    lowered = (first_char.lower() + remainder) if first_char.isupper() else stripped
    return sanitise_question_text(f"In this scene, {lowered}")


def add_bonus_questions(questions: List[Dict], rng: random.Random) -> List[Dict]:
    scenes: Dict[str, List[int]] = defaultdict(list)
    for idx, q in enumerate(questions):
        scene_id = q.get("source_scene_id")
        if scene_id:
            scenes[scene_id].append(idx)

    new_questions: List[Dict] = []
    for scene_id, idx_list in scenes.items():
        if not idx_list:
            continue
        chosen_idx = rng.choice(idx_list)
        original_question = questions[chosen_idx]
        clone = deepcopy(original_question)

        choice_labels = format_choice_labels(clone.get("choices"), clone.get("box_to_object"))
        clone["question"] = augment_question_text(clone.get("question", ""), choice_labels)
        if "original_question" in clone:
            clone["original_question"] = augment_original_question(clone.get("original_question"))

        clone.pop("question_index", None)
        new_questions.append(clone)

    updated = questions + new_questions

    for idx, question in enumerate(updated):
        question["question_index"] = idx

    return updated


def update_metadata(metadata_path: Path, questions: List[Dict]) -> None:
    if not metadata_path.exists():
        print(f"[WARN] Metadata file {metadata_path} not found; skipping metadata update.", file=sys.stderr)
        return

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to read metadata {metadata_path}: {exc}", file=sys.stderr)
        return

    type_counts = Counter(q.get("question_type") for q in questions if q.get("question_type"))
    category_counts = Counter(q.get("question_category") for q in questions if q.get("question_category"))
    scenes = {q.get("source_scene_id") for q in questions if q.get("source_scene_id")}

    metadata["total_questions"] = len(questions)
    metadata["question_type_counts"] = dict(sorted(type_counts.items()))
    metadata["question_category_counts"] = dict(sorted(category_counts.items()))
    if scenes:
        metadata["total_scenes"] = len(scenes)

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    questions = load_json(args.input)
    original_count = len(questions)
    updated_questions = add_bonus_questions(questions, rng)
    added = len(updated_questions) - original_count

    output_path = args.output if args.output else args.input
    save_json(output_path, updated_questions)

    if args.metadata:
        update_metadata(args.metadata, updated_questions)

    print(f"Scenes processed: {len({q.get('source_scene_id') for q in updated_questions if q.get('source_scene_id')})}")
    print(f"Original question count: {original_count}")
    print(f"Bonus questions added: {added}")
    print(f"New question count: {len(updated_questions)}")
    print(f"Output written to: {output_path}")
    if args.metadata:
        print(f"Metadata updated: {args.metadata}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Add one additional question per scene to an existing taxonomyQABench_simimage dataset.

For each scene (grouped by `source_scene_id`) we clone a randomly selected question,
lightly adjust the question text so it remains unique, and append it to the dataset.
The script recomputes question indices and optionally updates generation metadata.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append one additional random question per scene to sim image QA dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the existing all_questions.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to overwriting --input.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional generation_metadata.json path to update alongside the questions file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def invert_box_mapping(box_to_object: Optional[Dict[str, str]]) -> Dict[str, List[str]]:
    inverted: Dict[str, List[str]] = defaultdict(list)
    if isinstance(box_to_object, dict):
        for label, obj in box_to_object.items():
            if obj:
                inverted[obj].append(label)
    return inverted


def format_choice_labels(choices: Iterable[str], box_to_object: Optional[Dict[str, str]]) -> List[str]:
    inverted = invert_box_mapping(box_to_object)
    labels: List[str] = []
    for obj in choices or []:
        candidate_labels = inverted.get(obj)
        labels.append(candidate_labels[0] if candidate_labels else obj)
    return labels


def sanitise_question_text(question: str) -> str:
    """Normalise spacing around punctuation."""
    if not question:
        return question
    text = question.replace("  ", " ").strip()
    return text


def augment_question_text(question: str, choices: List[str]) -> str:
    """Prefix the question with 'In this scene,' to keep it unique."""
    if not isinstance(question, str) or not question.strip():
        return question

    stripped = question.strip()

    options_prefix = "Objects to choose from:"
    suffix = ""
    main = stripped
    if options_prefix in stripped:
        main, _, suffix = stripped.partition(options_prefix)
        main = main.strip()
        suffix = suffix.strip()

    if main.lower().startswith("in this scene"):
        augmented_main = main
    else:
        if main:
            first_char = main[0]
            remainder = main[1:] if len(main) > 1 else ""
            lowered = (first_char.lower() + remainder) if first_char.isupper() else main
        else:
            lowered = main
        augmented_main = f"In this scene, {lowered}"

    augmented_main = sanitise_question_text(augmented_main)

    if choices:
        formatted_choices = ", ".join(choices)
        full = f"{augmented_main} {options_prefix} {formatted_choices}".strip()
    else:
        full = augmented_main

    return sanitise_question_text(full)


def augment_original_question(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return text
    stripped = text.strip()
    if stripped.lower().startswith("in this scene"):
        return stripped
    first_char = stripped[0]
    remainder = stripped[1:] if len(stripped) > 1 else ""
    lowered = (first_char.lower() + remainder) if first_char.isupper() else stripped
    return f"In this scene, {lowered}"


def add_bonus_questions(questions: List[Dict], rng: random.Random) -> List[Dict]:
    scenes: Dict[str, List[int]] = defaultdict(list)
    for idx, q in enumerate(questions):
        scene_id = q.get("source_scene_id")
        if scene_id:
            scenes[scene_id].append(idx)

    new_questions: List[Dict] = []
    for scene_id, idx_list in scenes.items():
        if not idx_list:
            continue
        chosen_idx = rng.choice(idx_list)
        original_question = questions[chosen_idx]
        clone = deepcopy(original_question)

        # Rebuild choice labels and augment question strings
        choice_labels = format_choice_labels(clone.get("choices"), clone.get("box_to_object"))
        clone["question"] = augment_question_text(clone.get("question", ""), choice_labels)
        if "original_question" in clone and isinstance(clone["original_question"], str):
            clone["original_question"] = augment_original_question(clone["original_question"])

        clone.pop("question_index", None)  # will be reassigned later
        new_questions.append(clone)

    updated = questions + new_questions

    for idx, question in enumerate(updated):
        question["question_index"] = idx

    return updated


def update_metadata(metadata_path: Path, questions: List[Dict]) -> None:
    if not metadata_path.exists():
        print(f"[WARN] Metadata file {metadata_path} not found; skipping metadata update.", file=sys.stderr)
        return

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to read metadata {metadata_path}: {exc}", file=sys.stderr)
        return

    type_counts = Counter(q.get("question_type") for q in questions if q.get("question_type"))
    category_counts = Counter(q.get("question_category") for q in questions if q.get("question_category"))
    scenes = {q.get("source_scene_id") for q in questions if q.get("source_scene_id")}

    metadata["total_questions"] = len(questions)
    metadata["question_type_counts"] = dict(sorted(type_counts.items()))
    metadata["question_category_counts"] = dict(sorted(category_counts.items()))
    if scenes:
        metadata["total_scenes"] = len(scenes)

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    questions = load_json(args.input)
    original_count = len(questions)
    updated_questions = add_bonus_questions(questions, rng)
    added = len(updated_questions) - original_count

    output_path = args.output if args.output else args.input
    save_json(output_path, updated_questions)

    if args.metadata:
        update_metadata(args.metadata, updated_questions)

    print(f"Scenes processed: {len({q.get('source_scene_id') for q in updated_questions if q.get('source_scene_id')})}")
    print(f"Original question count: {original_count}")
    print(f"Bonus questions added: {added}")
    print(f"New question count: {len(updated_questions)}")
    print(f"Output written to: {output_path}")
    if args.metadata:
        print(f"Metadata updated: {args.metadata}")


if __name__ == "__main__":
    main()

