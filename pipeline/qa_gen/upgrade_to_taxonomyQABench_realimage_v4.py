#!/usr/bin/env python3
"""
upgrade taxonomy QA benchmarks to the v4 format.

Features:
1. Re-samples question phrasing using the template variants defined in
   `modules.qa_modules.question_templates.QuestionTemplates`, with deterministic
   randomness controlled by a seed.
2. Optionally augments each image with additional property-driven questions
   (`material_sound_absorption`, `material_thermal_touch`,
   `material_scratch_resistance`, and `physical_property`) with a configurable
   probability of inclusion per image/question type.
3. Emits an updated benchmark directory (`taxonomyQABench_realimage_v4`) that
   mirrors the source structure, updates metadata, and preserves original image
   assets via symlinks.

The script is designed to be deterministic and reproducible for auditability.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any, Set
from typing import Callable

LOGGER = logging.getLogger("upgrade_to_v4")

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_ROOT = SCRIPT_DIR / "scripts"
if MODULE_ROOT.exists():
    sys.path.insert(0, str(MODULE_ROOT))

try:
    from modules.qa_modules.question_templates import QuestionTemplates, FORCE_LAST_VARIANT_TYPES
    from modules.qa_modules.object_utils import ObjectUtils
    from modules.qa_modules.taxonomy_utils import TaxonomyUtils
    from modules.qa_modules.question_type_grouping import QuestionTypeGrouper
    from modules.qa_modules.filter_utils import is_void_cluster
    from modules.qa_modules.data_loading_utils import DataLoadingUtils
    from modules.qa_modules.cot_reasoning_utils import CoTReasoningGenerator
    from modules.qa_modules.legacy_question_type_utils import infer_legacy_question_type
    from modules.qa_modules.legacy_question_type_utils import infer_legacy_question_type
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Failed to import QA generation modules. Ensure this script resides in "
        "the qa_gen directory and that 'scripts/modules' is intact."
    ) from exc

def ensure_choice_clause(
    question_text: str,
    box_to_object: Optional[Dict[str, str]],
    question_type: Optional[str] = None,
) -> str:
    """
    Ensure the question string ends with 'Objects to choose from: <color labels>' built from box_to_object.
    """
    if not question_text:
        question_text = ""
    if not isinstance(box_to_object, dict) or not box_to_object:
        return question_text.strip()

    if question_type:
        normalized_type = question_type.lower()
        if normalized_type.startswith("spatial_") or normalized_type.startswith("manual_spatial"):
            return question_text.strip()

    color_labels = [label.strip() for label in box_to_object.keys() if label]
    if not color_labels:
        return question_text.strip()

    base = question_text.split("Objects to choose from:")[0].strip()
    if not base.endswith("?"):
        base = base.rstrip(".").rstrip()
        if not base.endswith("?"):
            base = f"{base}?"
    clause = f"Objects to choose from: {', '.join(color_labels)}"
    return f"{base} {clause}".strip()


def dedupe_questions_by_image_text(questions: List[Dict]) -> List[Dict]:
    """Remove duplicate entries that share both image_id and question string."""
    seen: Set[Tuple[Optional[str], str]] = set()
    deduped: List[Dict] = []
    for question in questions:
        image_id = question.get("image_id")
        text = (question.get("question") or "").strip()
        if not image_id or not text:
            deduped.append(question)
            continue
        key = (image_id, text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(question)
    return deduped


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_SEED = 4729
DEFAULT_APPEND_PROB = 0.2
PHYSICAL_PROPERTY_KEY = "physical_property"

PHYSICAL_PROPERTY_REASONING_TEMPLATE = (
    "I examine physical characteristics of each object to determine which "
    "exhibits {property_desc}. The {answer} clearly shows this property, "
    "while the remaining objects do not."
)

MATERIAL_REASONING_TEMPLATE = (
    "Considering the materials present in the scene, the {answer} best matches "
    "the expected {property_desc} property compared with the other objects."
)

HEURISTIC_TYPE_MAP = [
    ("rigid, movable, but not designed as a container", "compositional_set_subtraction_container"),
    ("repurposed as a reflector", "repurposing_reflector_concept"),
    ("repurposed as a container", "repurposing_container_concept"),
    ("repurposed as a cushion", "repurposing_cushion_concept"),
    ("repurposed as a shield", "repurposing_shield_concept"),
    ("repurposed as a stepstool", "repurposing_stepstool_concept"),
    ("repurposed as a bookend", "repurposing_bookend_concept"),
    ("repurposed as a lever", "repurposing_lever_concept"),
    ("most affected by high heat", "counterfactual_heat"),
    ("if water spills, which object gets damaged first", "counterfactual_water"),
    ("can be compressed to fit in tight spaces", "latent_compressible"),
    ("can be operated or used", "affordance_operate__use_device"),
    ("has the affordance of wearables_and_apparel", "affordance_wearables_and_apparel"),
    ("has the affordance of sit ride attend", "affordance_sit__ride__attend"),
    ("has the affordance of build span occupy", "affordance_build__span__occupy"),
    ("has the affordance of tableware_and_serveware", "affordance_tableware_and_serveware"),
    ("has the affordance of interact_with_living_moving_things", "affordance_interact_with_living_moving_things"),
    ("has the affordance of display exhibit signal_value", "affordance_display__exhibit__signal_value"),
    ("has the affordance of architectural_components_and_fixtures", "affordance_architectural_components_and_fixtures"),
    ("has the affordance of art_display (view_appraise)", "affordance_art_display_(view_appraise)"),
    ("has the affordance of structured_interaction", "affordance_structured_operational_engagement"),
    ("has the affordance of structured_operational_engagement", "affordance_structured_operational_engagement"),
    ("has the affordance of household facility_operations", "affordance_household__facility_operations"),
]

ALLOWED_CATEGORIES = {
    "taxonomy_description",
    "taxonomy_reasoning",
    "spatial_relation",
}

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
    question_text = question.get("question")
    if not isinstance(question_text, str):
        return
    if question.get("question_type") != "description_matching" and (
        question.get("question_category") != "taxonomy_description"
        or "matches this description" not in question_text
    ):
        return

    def _replace(match: re.Match) -> str:
        prefix, description, suffix = match.groups()
        cleaned = _clean_redundant_made_of(description)
        return f"{prefix}{cleaned}{suffix}"

    new_text = DESCRIPTION_PROMPT_REGEX.sub(_replace, question_text, count=1)
    question["question"] = new_text

    metadata = question.get("question_metadata")
    if isinstance(metadata, dict) and metadata.get("description"):
        metadata["description"] = _clean_redundant_made_of(metadata["description"])


# ---------------------------------------------------------------------------
# QA space helpers
# ---------------------------------------------------------------------------


def normalize_object_name(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def build_qa_space_valid_map(
    qa_space_data: Dict[str, Any],
    taxonomy_utils: Optional[TaxonomyUtils] = None,
) -> Dict[str, Set[str]]:
    mappings = qa_space_data.get("question_answer_mappings", {}) if qa_space_data else {}
    normalized: Dict[str, Set[str]] = {}
    for question_type, data in mappings.items():
        valid: Set[str] = set()
        for key in ("sm_valid_objects", "openimages_valid_objects"):
            for obj in data.get(key, []) or []:
                norm = normalize_object_name(obj)
                if norm:
                    valid.add(norm)
        if valid:
            normalized[question_type] = valid

    if taxonomy_utils and "physical_property" not in normalized:
        physical_taxonomy = taxonomy_utils.taxonomy_clusters.get("final_taxonomy_physical_properties", {})
        property_clusters = physical_taxonomy.get("physical_properties", {}) if physical_taxonomy else {}
        physical_objects: Set[str] = set()
        for cluster_data in property_clusters.values():
            for obj in cluster_data.get("objects", []) or []:
                norm = normalize_object_name(obj)
                if norm:
                    physical_objects.add(norm)
        if physical_objects:
            normalized["physical_property"] = physical_objects

    return normalized


def filter_objects_for_question(
    objects: Iterable[str],
    valid_set: Optional[Set[str]] = None,
    ensure_inclusion: Optional[str] = None,
    predicate: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    filtered: List[str] = []
    seen: Set[str] = set()

    for obj in objects:
        if not obj:
            continue
        norm = normalize_object_name(obj)
        if not norm or norm in seen:
            continue
        if valid_set is not None and norm not in valid_set:
            continue
        if predicate and not predicate(obj):
            continue
        filtered.append(obj)
        seen.add(norm)

    if ensure_inclusion:
        norm = normalize_object_name(ensure_inclusion)
        if norm and norm not in seen:
            filtered.append(ensure_inclusion)

    return filtered


def get_box_label(box_to_object: Dict[str, str], object_name: str) -> Optional[str]:
    for box_label, obj in box_to_object.items():
        if obj == object_name:
            return box_label
    return None


def build_choice_labels(objects: Iterable[str], box_to_object: Optional[Dict[str, str]]) -> List[str]:
    ordered_objects = [obj for obj in objects if obj]
    if not box_to_object:
        seen: Set[str] = set()
        labels: List[str] = []
        for obj in ordered_objects:
            if obj not in seen:
                labels.append(obj)
                seen.add(obj)
        return labels

    inverse: Dict[str, List[str]] = defaultdict(list)
    for box_label, obj_name in box_to_object.items():
        if obj_name:
            inverse[obj_name].append(box_label)

    labels: List[str] = []
    seen: Set[str] = set()
    for obj in ordered_objects:
        label_list = inverse.get(obj)
        if label_list:
            label = label_list[0]
        else:
            label = obj
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def normalize_question_choices(question: Dict) -> None:
    qtype = question.get("question_type")
    if isinstance(qtype, str) and qtype.startswith("spatial_"):
        return
    box_map = question.get("box_to_object")
    if not isinstance(box_map, dict) or not box_map:
        return

    inverse: Dict[str, List[str]] = defaultdict(list)
    for box_label, obj_name in box_map.items():
        if obj_name:
            inverse[obj_name].append(box_label)

    object_list = [obj for obj in question.get("objects", []) if obj in inverse]

    if not object_list:
        # Fall back to all mapped objects if objects list is empty or mismatched.
        object_list = list(dict.fromkeys(filter(None, box_map.values())))

    choice_labels = build_choice_labels(object_list, box_map)

    answer = question.get("answer")
    if answer not in choice_labels:
        target = question.get("target_object")
        resolved_label = None
        if target:
            resolved_label = get_box_label(box_map, target)
        if not resolved_label and answer:
            resolved_label = get_box_label(box_map, answer)
        if resolved_label:
            question["answer"] = resolved_label
            answer = resolved_label
        if answer and answer not in choice_labels:
            choice_labels.append(answer)

    question["choices"] = choice_labels
    question["objects"] = object_list


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def build_template_patterns(question_templates: QuestionTemplates) -> Dict[str, List[re.Pattern]]:
    """Compile regex patterns for each template with named capture groups."""
    placeholder_regex = re.compile(r"{([^}]+)}")
    patterns: Dict[str, List[re.Pattern]] = {}

    for q_type, template_values in question_templates.templates.items():
        variants = (
            template_values
            if isinstance(template_values, list)
            else [template_values]
        )
        compiled_variants: List[re.Pattern] = []

        for variant in variants:
            # Skip non-string templates defensively.
            if not isinstance(variant, str):
                continue

            parts: List[str] = []
            last_idx = 0
            for match in placeholder_regex.finditer(variant):
                start, end = match.span()
                placeholder_name = match.group(1)

                literal_segment = re.escape(variant[last_idx:start])
                parts.append(literal_segment)
                parts.append(f"(?P<{placeholder_name}>.+?)")
                last_idx = end

            parts.append(re.escape(variant[last_idx:]))

            regex_body = "".join(parts)
            regex_body = regex_body.replace(r"\ ", r"\s+")
            pattern = re.compile(f"^{regex_body}$", re.IGNORECASE)
            compiled_variants.append(pattern)

        patterns[q_type] = compiled_variants

    return patterns


def determine_category_for_type(
    question_type: Optional[str],
    existing_category: Optional[str],
    grouper: QuestionTypeGrouper,
) -> str:
    """
    Map a detailed question_type to one of the consolidated categories.
    """
    if question_type:
        simplified = grouper.get_simplified_type(question_type)
        if simplified in ALLOWED_CATEGORIES:
            return simplified

    if existing_category in ALLOWED_CATEGORIES:
        return existing_category

    if question_type:
        if question_type.startswith("spatial_") or question_type == "legacy_spatial_relation":
            return "spatial_relation"
        if question_type.startswith(
            (
                "repurposing_",
                "counterfactual_",
                "latent_",
                "functional_",
                "compositional_",
                "taxonomy_reasoning",
                "legacy_taxonomy_reasoning",
            )
        ):
            return "taxonomy_reasoning"

    return "taxonomy_description"


def infer_question_type(
    question: Dict,
    template_patterns: Dict[str, List[re.Pattern]]
) -> Tuple[Optional[str], Dict[str, str]]:
    """Infer question type from question text using template patterns."""
    base_text = (question.get("original_question") or question.get("question") or "").strip()
    if not base_text:
        return None, {}

    for q_type, compiled_variants in template_patterns.items():
        for pattern in compiled_variants:
            match = pattern.match(base_text)
            if match:
                extracted = {k: v.strip().strip("'\"") for k, v in match.groupdict().items() if v}
                return q_type, extracted
    return None, {}


def choose_template_variant(
    rng: random.Random,
    templates: QuestionTemplates,
    question_type: str
) -> Optional[str]:
    """Select a deterministic template variant for a given question type."""
    template_value = templates.templates.get(question_type)
    if template_value is None:
        return None
    if isinstance(template_value, list):
        if not template_value:
            return None
        if question_type in FORCE_LAST_VARIANT_TYPES:
            return template_value[-1]
        return rng.choice(template_value)
    if isinstance(template_value, str):
        return template_value
    return None


def rephrase_question(
    rng: random.Random,
    templates: QuestionTemplates,
    question: Dict,
    question_type: str,
    placeholder_values: Dict[str, str]
) -> Optional[str]:
    """
    Re-sample the question text using template variants.

    Returns the updated question text, or None if the operation fails.
    """
    new_template = choose_template_variant(rng, templates, question_type)
    if not new_template:
        return None

    placeholders_in_template = set(re.findall(r"{([^}]+)}", new_template))

    format_kwargs = dict(placeholder_values)
    if question_type == "physical_property" and PHYSICAL_PROPERTY_KEY in placeholder_values:
        format_kwargs.setdefault("property", placeholder_values[PHYSICAL_PROPERTY_KEY])

    if placeholders_in_template:
        missing_keys = {key for key in placeholders_in_template if key not in format_kwargs}
        if missing_keys:
            LOGGER.debug(
                "Missing placeholders %s for question_type=%s; skipping rephrase.",
                missing_keys,
                question_type,
            )
            return None
        try:
            return new_template.format(**format_kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "Formatting failed for question_type=%s with template=%s: %s",
                question_type,
                new_template,
                exc,
            )
            return None

    # Template requires no placeholders.
    return new_template


def collect_scene_objects(questions: Iterable[Dict]) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-image objects and box mappings from existing questions."""
    scene_data: Dict[str, Dict[str, Any]] = {}

    for q in questions:
        image_id = q.get("image_id")
        if not image_id:
            continue

        data = scene_data.setdefault(
            image_id,
            {"objects": set(), "box_to_object": {}},
        )

        box_map = q.get("box_to_object")
        if isinstance(box_map, dict):
            for box_label, obj_name in box_map.items():
                if not obj_name:
                    continue
                data["box_to_object"].setdefault(box_label, obj_name)
                data["objects"].add(obj_name)
            continue

        for bucket in ("objects", "choices"):
            for obj in q.get(bucket, []) or []:
                if obj:
                    data["objects"].add(obj)

    return scene_data


def _is_valid_material_candidate(
    candidate: str,
    objects: Iterable[str],
    taxonomy_utils: TaxonomyUtils,
) -> bool:
    clusters = taxonomy_utils.get_object_clusters(candidate, "final_taxonomy_material") or []
    clusters = [c for c in clusters if not is_void_cluster(c, "material")]
    if not clusters:
        return False
    for other in objects:
        if other == candidate:
            continue
        other_clusters = taxonomy_utils.get_object_clusters(other, "final_taxonomy_material") or []
        if any(cluster in other_clusters for cluster in clusters):
            return False
    return True


def _is_valid_physical_candidate(
    candidate: str,
    property_name: str,
    objects: Iterable[str],
    object_utils: ObjectUtils,
) -> bool:
    candidate_props = object_utils.get_object_physical_properties(candidate)
    if property_name not in candidate_props:
        return False
    for other in objects:
        if other == candidate:
            continue
        if property_name in object_utils.get_object_physical_properties(other):
            return False
    return True


def select_material_candidate(
    rng: random.Random,
    objects: Iterable[str],
    question_type: str,
    qa_space_valid_objects: Optional[Set[str]] = None,
) -> Optional[str]:
    """Select an object strictly from the QA-space valid set for this material question."""

    if not qa_space_valid_objects:
        LOGGER.debug("QA space missing for %s; skipping material candidate selection.", question_type)
        return None

    eligible: List[str] = []
    for obj_name in objects:
        norm = normalize_object_name(obj_name)
        if norm in qa_space_valid_objects:
            eligible.append(obj_name)

    if not eligible:
        return None

    return rng.choice(eligible)


def select_physical_property_candidate(
    rng: random.Random,
    objects: Iterable[str],
    object_utils: ObjectUtils,
    qa_space_valid_objects: Optional[Set[str]] = None,
) -> Optional[Tuple[str, str]]:
    """Select an object with a known physical property constrained by QA space."""

    if qa_space_valid_objects is None:
        LOGGER.debug("QA space missing for physical_property; skipping candidate selection.")
        return None

    candidates: List[Tuple[str, str]] = []
    for obj_name in objects:
        norm = normalize_object_name(obj_name)
        if norm not in qa_space_valid_objects:
            continue
        properties = object_utils.get_object_physical_properties(obj_name)
        if properties:
            property_choice = rng.choice(properties)
            candidates.append((obj_name, property_choice))
    if not candidates:
        return None
    return rng.choice(candidates)


def ensure_symlink(target: Path, link_path: Path) -> None:
    """Create or refresh a symlink."""
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target, target_is_directory=True)


def update_scene_statistics(questions: List[Dict]) -> List[Dict]:
    """Recalculate scene statistics from questions."""
    counts: Dict[str, int] = defaultdict(int)
    for q in questions:
        image_id = q.get("image_id")
        if image_id:
            counts[image_id] += 1
    stats = [{"scene_id": img, "image_id": img, "question_count": cnt} for img, cnt in counts.items()]
    stats.sort(key=lambda entry: entry["image_id"])
    return stats


def recompute_type_statistics(questions: List[Dict]) -> Dict[str, int]:
    """Aggregate question type counts."""
    counter: Counter = Counter()
    for q in questions:
        q_type = q.get("question_type", "unknown")
        counter[q_type] += 1
    return dict(counter)


def recompute_category_statistics(questions: List[Dict]) -> Dict[str, int]:
    """Aggregate question category counts."""
    counter: Counter = Counter()
    for q in questions:
        cat = q.get("question_category", "unknown")
        counter[cat] += 1
    return dict(counter)


# ---------------------------------------------------------------------------
# Core processing routines
# ---------------------------------------------------------------------------

def retokenize_existing_questions(
    rng: random.Random,
    templates: QuestionTemplates,
    template_patterns: Dict[str, List[re.Pattern]],
    questions: List[Dict],
    grouper: QuestionTypeGrouper,
    object_utils: ObjectUtils,
    taxonomy_utils: TaxonomyUtils
) -> None:
    """Randomly rephrase each question using available template variants."""
    for question in questions:
        # Preserve manually curated questions exactly as provided
        if question.get("manual_entry") or question.get("is_manual"):
            if not question.get("question_type"):
                category_slug = (question.get("question_category") or "manual").strip().lower().replace(" ", "_")
                question["question_type"] = f"manual_{category_slug}"
            category_value = question.get("question_category")
            if category_value not in ALLOWED_CATEGORIES:
                question["question_category"] = determine_category_for_type(
                    question.get("question_type"),
                    category_value,
                    grouper,
                )
            question.setdefault("rephrased_with_template", False)
            continue
        if question.get("legacy_entry"):
            if not question.get("question_type"):
                legacy_type = infer_legacy_question_type(question)
                if legacy_type:
                    question["question_type"] = legacy_type
            question["question_category"] = determine_category_for_type(
                question.get("question_type"),
                question.get("question_category"),
                grouper,
            )
            question.setdefault("rephrased_with_template", False)
            continue

        # Preserve existing question_type if present, otherwise try to infer it
        existing_type = question.get("question_type")
        if not existing_type:
            legacy_type = infer_legacy_question_type(question)
            if legacy_type:
                question["question_type"] = legacy_type
                question["question_category"] = determine_category_for_type(
                    legacy_type,
                    question.get("question_category"),
                    grouper,
                )
                existing_type = legacy_type
        inferred_type, placeholders = infer_question_type(question, template_patterns)

        question_type = existing_type or inferred_type

        if not question_type:
            apply_question_type_heuristics(question, grouper)
            question_type = question.get("question_type")
            if not question_type:
                continue

        # Set question_type if it wasn't already set
        if not existing_type and inferred_type:
            question["question_type"] = inferred_type

        # Ensure question_category is always set based on question_type
        question["question_category"] = determine_category_for_type(
            question_type,
            question.get("question_category"),
            grouper,
        )

        # Attempt to extract placeholders from existing question text when absent.
        if not placeholders:
            placeholders = extract_placeholders_from_text(question_type, question)

        placeholders = ensure_placeholder_values(
            question_type,
            question,
            placeholders,
            object_utils,
            taxonomy_utils,
        )

        updated_text = rephrase_question(
            rng=rng,
            templates=templates,
            question=question,
            question_type=question_type,
            placeholder_values=placeholders,
        )

        if updated_text:
            if "original_question" not in question:
                question["original_question"] = question.get("question", updated_text)
            question["question"] = updated_text
            question.setdefault("rephrased_with_template", True)
        else:
            question.setdefault("rephrased_with_template", False)

        apply_question_type_heuristics(question, grouper)
        if question.get("box_to_object") and not question.get("manual_entry"):
            question["question"] = ensure_choice_clause(
                question.get("question", ""),
                question.get("box_to_object"),
                question_type=question_type,
            )
        normalize_question_choices(question)
        sanitize_description_question(question)


def extract_placeholders_from_text(question_type: str, question: Dict) -> Dict[str, str]:
    """Best-effort extraction of placeholder values from an existing question."""
    text = question.get("original_question") or question.get("question") or ""
    text_lower = text.lower()
    values: Dict[str, str] = {}

    if question_type == "material_property":
        match = re.search(r"made of ['\"]?([^'\"?]+)", text, flags=re.IGNORECASE)
        if match:
            values["material"] = match.group(1).strip()
    elif question_type == "function_knowledge":
        match = re.search(r"(?:used as|designed for)\s+['\"]?([^'\"?]+)", text, flags=re.IGNORECASE)
        if match:
            values["function"] = match.group(1).strip()
    elif question_type == "description_matching":
        match = re.search(r"description[: ]+['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
        if match:
            values["description"] = match.group(1).strip()
    elif question_type == "physical_property":
        match = re.search(r"physical (?:property|trait) of ['\"]?([^'\"?]+)", text, flags=re.IGNORECASE)
        if match:
            values[PHYSICAL_PROPERTY_KEY] = match.group(1).strip()
    return values


def ensure_placeholder_values(
    question_type: str,
    question: Dict,
    placeholders: Dict[str, str],
    object_utils: ObjectUtils,
    taxonomy_utils: TaxonomyUtils,
) -> Dict[str, str]:
    answer_object = (
        question.get("original_answer")
        or question.get("target_object")
        or question.get("answer")
    )

    if question_type == "material_property" and "material" not in placeholders:
        if answer_object:
            material = object_utils.get_object_material(answer_object)
            if material:
                placeholders["material"] = material

    if question_type == "function_knowledge" and "function" not in placeholders:
        if answer_object:
            function = object_utils.get_object_function(answer_object)
            if function:
                placeholders["function"] = function

    if question_type == "description_matching" and "description" not in placeholders:
        description = question.get("question_metadata", {}).get("description")
        if not description and answer_object:
            description = object_utils.get_object_description(answer_object)
        if description:
            placeholders["description"] = description

    if question_type == "physical_property" and PHYSICAL_PROPERTY_KEY not in placeholders:
        property_name = question.get("question_metadata", {}).get(PHYSICAL_PROPERTY_KEY)
        if not property_name and answer_object:
            props = object_utils.get_object_physical_properties(answer_object)
            if props:
                property_name = props[0]
        if property_name:
            placeholders[PHYSICAL_PROPERTY_KEY] = property_name

    return placeholders


def augment_with_property_questions(
    rng: random.Random,
    templates: QuestionTemplates,
    questions: Iterable[Dict],
    scene_objects: Dict[str, Dict[str, Any]],
    object_utils: ObjectUtils,
    taxonomy_utils: TaxonomyUtils,
    grouper: QuestionTypeGrouper,
    append_probability: float,
    starting_index: int,
    qa_space_valid_map: Optional[Dict[str, Set[str]]] = None,
    cot_generator: Optional[CoTReasoningGenerator] = None,
) -> Tuple[List[Dict], int]:
    """Generate new property-focused questions per image."""
    new_questions: List[Dict] = []
    next_index = starting_index

    existing_questions_by_image: Dict[str, Set[str]] = defaultdict(set)
    for question in questions:
        image_id = question.get("image_id")
        text = question.get("question")
        if image_id and text:
            existing_questions_by_image[image_id].add(text.strip())

    material_descriptors = {
        "material_sound_absorption": "sound absorptive or fabric-based",
        "material_thermal_touch": "materials that feel cold to the touch",
        "material_scratch_resistance": "materials resistant to scratching",
    }

    qa_space_physical_objects: Optional[Set[str]] = None
    if qa_space_valid_map:
        qa_space_physical_objects = qa_space_valid_map.get("physical_property")

    for image_id, scene_info in scene_objects.items():
        scene_objects_set = set(scene_info.get("objects", set())) if scene_info else set()
        box_to_object = scene_info.get("box_to_object", {}) if scene_info else {}
        box_objects = {obj for obj in box_to_object.values() if obj}

        object_candidates = box_objects or scene_objects_set

        if not object_candidates:
            continue

        # Material inference questions.
        for question_type, descriptor in material_descriptors.items():
            if rng.random() >= append_probability:
                continue
            valid_objects_for_type = None
            if qa_space_valid_map:
                valid_objects_for_type = qa_space_valid_map.get(question_type)
            if not valid_objects_for_type:
                continue
            candidate = select_material_candidate(
                rng=rng,
                objects=object_candidates,
                question_type=question_type,
                qa_space_valid_objects=valid_objects_for_type,
            )
            if not candidate:
                continue

            if not _is_valid_material_candidate(candidate, object_candidates, taxonomy_utils):
                continue

            template = choose_template_variant(rng, templates, question_type)
            if not template:
                continue

            question_text = template
            final_question_text = ensure_choice_clause(
                question_text,
                box_to_object,
                question_type=question_type,
            )
            if final_question_text in existing_questions_by_image[image_id]:
                continue
            choice_objects = sorted(set(object_candidates))
            if candidate not in choice_objects:
                choice_objects.append(candidate)
            choice_objects = sorted(set(choice_objects))

            if cot_generator:
                reasoning = cot_generator.generate_comprehensive_reasoning(
                    question_type=question_type,
                    target_object=candidate,
                    object_set=list(choice_objects),
                    answer=candidate,
                    qa_filter_descriptor=descriptor,
                    material=descriptor,
                    box_to_object=box_to_object,
                )
            else:
                reasoning = MATERIAL_REASONING_TEMPLATE.format(
                    answer=candidate,
                    property_desc=descriptor,
                )

            answer_label = get_box_label(box_to_object, candidate) or candidate
            choice_labels = build_choice_labels(choice_objects, box_to_object)

            if answer_label not in choice_labels:
                choice_labels.append(answer_label)

            new_entry = build_question_entry(
                question_type=question_type,
                question_text=final_question_text,
                answer=answer_label,
                target_object=candidate,
                image_id=image_id,
                objects=choice_objects,
                choices=choice_labels,
                reasoning=reasoning,
                question_index=next_index,
                grouper=grouper,
                box_to_object=box_to_object,
            )
            existing_questions_by_image[image_id].add(final_question_text)
            normalize_question_choices(new_entry)
            new_questions.append(new_entry)
            next_index += 1

        # Physical property questions.
        if qa_space_physical_objects and rng.random() < append_probability:
            physical_candidate = select_physical_property_candidate(
                rng=rng,
                objects=object_candidates,
                object_utils=object_utils,
                qa_space_valid_objects=qa_space_physical_objects,
            )
            if physical_candidate:
                candidate_object, property_name = physical_candidate
                if not _is_valid_physical_candidate(candidate_object, property_name, object_candidates, object_utils):
                    continue
                template = choose_template_variant(rng, templates, "physical_property")
                if template:
                    try:
                        question_text = template.format(physical_property=property_name, property=property_name)
                    except KeyError:
                        question_text = template.replace("{property}", property_name)
                    final_question_text = ensure_choice_clause(
                        question_text,
                        box_to_object,
                        question_type="physical_property",
                    )
                    if final_question_text in existing_questions_by_image[image_id]:
                        continue
                    # Ensure answer object has a valid physical-property cluster (non-void)
                    clusters = taxonomy_utils.get_object_clusters(candidate_object, 'final_taxonomy_physical_properties') or []
                    valid_clusters = [
                        c for c in clusters
                        if c and not is_void_cluster(c, 'physical')
                    ]
                    if not valid_clusters:
                        continue
                    choice_objects = sorted(set(object_candidates))
                    if candidate_object not in choice_objects:
                        choice_objects.append(candidate_object)
                    choice_objects = sorted(set(choice_objects))

                    if cot_generator:
                        reasoning = cot_generator.generate_comprehensive_reasoning(
                            question_type="physical_property",
                            target_object=candidate_object,
                            object_set=list(choice_objects),
                            answer=candidate_object,
                            physical_property=property_name,
                            qa_filter_descriptor=property_name,
                            box_to_object=box_to_object,
                        )
                    else:
                        reasoning = PHYSICAL_PROPERTY_REASONING_TEMPLATE.format(
                            answer=candidate_object,
                            property_desc=property_name,
                        )

                    answer_label = get_box_label(box_to_object, candidate_object) or candidate_object
                    choice_labels = build_choice_labels(choice_objects, box_to_object)

                    if answer_label not in choice_labels:
                        choice_labels.append(answer_label)
                    new_entry = build_question_entry(
                        question_type="physical_property",
                        question_text=final_question_text,
                        answer=answer_label,
                        target_object=candidate_object,
                        image_id=image_id,
                        objects=choice_objects,
                        choices=choice_labels,
                        reasoning=reasoning,
                        question_index=next_index,
                        grouper=grouper,
                        box_to_object=box_to_object,
                        extra_placeholders={PHYSICAL_PROPERTY_KEY: property_name},
                    )
                    existing_questions_by_image[image_id].add(final_question_text)
                    normalize_question_choices(new_entry)
                    new_questions.append(new_entry)
                    next_index += 1

    return new_questions, next_index


def build_question_entry(
    question_type: str,
    question_text: str,
    answer: str,
    target_object: Optional[str],
    image_id: str,
    objects: Iterable[str],
    choices: Iterable[str],
    reasoning: str,
    question_index: int,
    grouper: QuestionTypeGrouper,
    box_to_object: Optional[Dict[str, str]] = None,
    extra_placeholders: Optional[Dict[str, str]] = None,
) -> Dict:
    """Build a question dictionary aligned with existing dataset structure."""
    object_list = sorted({obj for obj in objects if obj})
    choice_list = list(dict.fromkeys(label for label in choices if label))
    if answer not in choice_list:
        choice_list.append(answer)

    entry = {
        "question": ensure_choice_clause(question_text, box_to_object, question_type=question_type),
        "answer": answer,
        "target_object": target_object or answer,
        "objects": object_list,
        "choices": choice_list,
        "reasoning": reasoning,
        "image_path": f"{image_id}/bbox.jpg",
        "image_id": image_id,
        "question_index": question_index,
        "question_type": question_type,
        "question_category": determine_category_for_type(
            question_type,
            None,
            grouper,
        ),
        "original_question": question_text,
        "original_answer": answer,
    }

    if box_to_object:
        entry["box_to_object"] = dict(box_to_object)

    if extra_placeholders:
        entry["question_metadata"] = extra_placeholders

    sanitize_description_question(entry)

    return entry


def refresh_metadata_files(
    output_dir: Path,
    questions: List[Dict],
    scene_stats: List[Dict],
    question_type_counts: Dict[str, int],
    question_category_counts: Dict[str, int],
) -> None:
    """Persist summary metadata files."""
    total_questions = len(questions)
    unique_images = len({q.get("image_id") for q in questions if q.get("image_id")})

    qtype_stats_path = output_dir / "question_type_statistics.json"
    new_qtype_stats = {
        "total_questions": total_questions,
        "unique_qa_pairs": total_questions,
        "question_category_counts": question_category_counts,
        "question_type_counts": question_type_counts,
        "unique_question_categories": len(question_category_counts),
        "unique_images": unique_images,
        "note": "Statistics regenerated by upgrade_to_taxonomyQABench_realimage_v4.py",
    }
    qtype_stats_path.write_text(json.dumps(new_qtype_stats, indent=2))

    scene_stats_path = output_dir / "scene_statistics.json"
    scene_stats_path.write_text(json.dumps(scene_stats, indent=2))

    summary_path = output_dir / "combining_summary.json"
    summary_payload = {
        "source_benchmarks": {
            "upgraded_from": str(output_dir.name),
        },
        "total_unique_questions": total_questions,
        "total_unique_images": unique_images,
        "augmentation": {
            "material_sound_absorption": question_type_counts.get("material_sound_absorption", 0),
            "material_thermal_touch": question_type_counts.get("material_thermal_touch", 0),
            "material_scratch_resistance": question_type_counts.get("material_scratch_resistance", 0),
            "physical_property": question_type_counts.get("physical_property", 0),
        },
        "note": "Auto-generated by upgrade_to_taxonomyQABench_realimage_v4.py",
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    metadata_path = output_dir / "generation_metadata.json"
    metadata_payload = {
        "generation_info": {
            "total_questions": total_questions,
            "total_scenes": unique_images,
        },
        "question_type_counts": question_type_counts,
        "question_category_counts": question_category_counts,
        "note": "Simplified metadata regenerated by upgrade_to_taxonomyQABench_realimage_v4.py",
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))


# ---------------------------------------------------------------------------
# Main execution flow
# ---------------------------------------------------------------------------


def apply_question_type_heuristics(question: Dict, grouper: QuestionTypeGrouper) -> None:
    """Assign question_type using heuristic string matching when templating fails."""
    text = (question.get("original_question") or question.get("question") or "").lower()
    if "matches this description" in text:
        question["question_type"] = "description_matching"
        question["question_category"] = determine_category_for_type(
            "description_matching",
            question.get("question_category"),
            grouper,
        )
        return
    if "is made of" in text:
        question["question_type"] = "material_property"
        question["question_category"] = determine_category_for_type(
            "material_property",
            question.get("question_category"),
            grouper,
        )
        return
    if "is used as" in text:
        question["question_type"] = "function_knowledge"
        question["question_category"] = determine_category_for_type(
            "function_knowledge",
            question.get("question_category"),
            grouper,
        )
        return
    for substring, q_type in HEURISTIC_TYPE_MAP:
        if substring in text:
            question["question_type"] = q_type
            question["question_category"] = determine_category_for_type(
                q_type,
                question.get("question_category"),
                grouper,
            )
            return
    if "question_type" not in question:
        question["question_type"] = "unknown"
    if "question_category" not in question:
        question["question_category"] = determine_category_for_type(
            question.get("question_type"),
            question.get("question_category"),
            grouper,
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upgrade taxonomy QA benchmark to v4.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/path/to/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_final"),
        help="Path to the source benchmark directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("taxonomyQABench_realimage_final_polished"),
        help="Destination directory for the upgraded benchmark (relative to qa_gen or absolute).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic template sampling.",
    )
    parser.add_argument(
        "--append-probability",
        type=float,
        default=DEFAULT_APPEND_PROB,
        help="Probability of adding each supplemental property question per image.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    rng = random.Random(args.seed)

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir if args.output_dir.is_absolute() else (SCRIPT_DIR / args.output_dir)).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input benchmark directory not found: {input_dir}")

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    # Symlink the images directory to avoid duplicating assets.
    input_images_dir = input_dir / "images"
    output_images_dir = output_dir / "images"
    if input_images_dir.exists():
        ensure_symlink(input_images_dir, output_images_dir)
        LOGGER.info("Linked images directory to %s", output_images_dir)
    else:
        LOGGER.warning("Input images directory missing; skipping symlink.")

    # Load questions.
    all_questions_path = input_dir / "all_questions.json"
    with all_questions_path.open("r") as f:
        questions = json.load(f)

    # Prepare utilities.
    templates = QuestionTemplates()
    template_patterns = build_template_patterns(templates)
    grouper = QuestionTypeGrouper()

    taxonomy_dir = SCRIPT_DIR / "taxonomy"
    taxonomy_utils = TaxonomyUtils(taxonomy_dir)
    object_utils = ObjectUtils(taxonomy_utils=taxonomy_utils)
    data_loader = DataLoadingUtils()
    qa_space_data = data_loader.load_qa_space_data()
    qa_space_valid_map = build_qa_space_valid_map(qa_space_data, taxonomy_utils=taxonomy_utils)
    cot_generator = CoTReasoningGenerator(taxonomy_utils=taxonomy_utils)

    # Step 1: rephrase existing questions.
    retokenize_existing_questions(
        rng=rng,
        templates=templates,
        template_patterns=template_patterns,
        questions=questions,
        grouper=grouper,
        object_utils=object_utils,
        taxonomy_utils=taxonomy_utils,
    )

    # Collect scene objects for augmentation.
    scene_objects = collect_scene_objects(questions)
    max_index = max((q.get("question_index", -1) for q in questions), default=-1) + 1

    # Load QA space valid map
    # qa_space_valid_map = None # This line is removed as it's now loaded in main
    # qa_space_data_path = input_dir / "question_answer_mappings.json" # This line is removed as it's now loaded in main
    # if qa_space_data_path.exists(): # This line is removed as it's now loaded in main
    #     with qa_space_data_path.open("r") as f: # This line is removed as it's now loaded in main
    #         qa_space_data = json.load(f) # This line is removed as it's now loaded in main
    #     qa_space_valid_map = build_qa_space_valid_map(qa_space_data) # This line is removed as it's now loaded in main
    #     LOGGER.info("Loaded QA space valid map from %s", qa_space_data_path) # This line is removed as it's now loaded in main
    # else: # This line is removed as it's now loaded in main
    #     LOGGER.warning("QA space valid map not found at %s; property questions will not be filtered.", qa_space_data_path) # This line is removed as it's now loaded in main

    new_questions, next_index = augment_with_property_questions(
        rng=rng,
        templates=templates,
        questions=questions,
        scene_objects=scene_objects,
        object_utils=object_utils,
        taxonomy_utils=taxonomy_utils,
        grouper=grouper,
        append_probability=args.append_probability,
        starting_index=max_index,
        qa_space_valid_map=qa_space_valid_map,
        cot_generator=cot_generator,
    )

    # Append new questions and write updated all_questions file.
    original_count = len(questions)
    questions.extend(new_questions)
    questions = dedupe_questions_by_image_text(questions)
    added_count = len(questions) - original_count
    (output_dir / "all_questions.json").write_text(json.dumps(questions, indent=2))
    LOGGER.info("Wrote %d questions to %s", len(questions), output_dir / "all_questions.json")

    # Update metadata assets.
    scene_stats = update_scene_statistics(questions)
    question_type_counts = recompute_type_statistics(questions)
    question_category_counts = recompute_category_statistics(questions)

    refresh_metadata_files(
        output_dir=output_dir,
        questions=questions,
        scene_stats=scene_stats,
        question_type_counts=question_type_counts,
        question_category_counts=question_category_counts,
    )

    LOGGER.info(
        "Upgrade complete. New benchmark written to %s (added %d property questions).",
        output_dir,
        added_count,
    )


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()


