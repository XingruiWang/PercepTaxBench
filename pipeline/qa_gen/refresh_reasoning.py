#!/usr/bin/env python3
import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.qa_modules.taxonomy_utils import TaxonomyUtils
from modules.qa_modules.object_utils import ObjectUtils
from modules.qa_modules.cot_reasoning_utils import CoTReasoningGenerator

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate reasoning strings for an existing all_questions.json using current"
            " Chain-of-Thought templates and taxonomy utilities."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to all_questions.json to refresh.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output file. If omitted, you must pass --in-place to overwrite the input file.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file with refreshed reasoning (mutually exclusive with --output).",
    )
    parser.add_argument(
        "--taxonomy-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "taxonomy",
        help="Directory containing taxonomy JSON files (defaults to qa_gen/taxonomy).",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=None,
        help="Directory containing annotation JSON files for spatial reasoning. "
             "If omitted, the script attempts to infer the correct path from generation metadata.",
    )
    parser.add_argument(
        "--reasoning-output",
        type=Path,
        help="Optional path to write reasoning records separately (question_index + reasoning).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def _resolve_answer_object(question: Dict[str, Any]) -> Optional[str]:
    box_map = question.get("box_to_object") or {}
    answer_label = question.get("answer")
    if box_map and answer_label in box_map:
        return box_map[answer_label]
    if question.get("target_object"):
        return question["target_object"]
    return answer_label


def _resolve_object_list(question: Dict[str, Any]) -> List[str]:
    box_map = question.get("box_to_object") or {}
    choices = question.get("choices") or question.get("objects") or []
    if not choices:
        return []
    resolved: List[str] = []
    if box_map:
        for label in choices:
            resolved.append(box_map.get(label, label))
    else:
        resolved = list(choices)
    return [obj for obj in resolved if obj]


def _sanitize_box_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return label
    cleaned = re.sub(r"\s+", " ", label).strip()
    if "object in" in cleaned.lower():
        parts = re.split(r"object in", cleaned, flags=re.IGNORECASE)
        tail = parts[-1].strip()
        if tail:
            cleaned = tail
        else:
            cleaned = parts[0].strip()
    return cleaned


def _compose_spatial_phrase(label: Optional[str], name: Optional[str]) -> Optional[str]:
    label = _sanitize_box_label(label)
    if label and name:
        return f"the {name} in {label}"
    if name:
        return f"the {name}"
    if label:
        return f"the object in {label}"
    return None


def _ensure_dict(value: Any, default_key: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {default_key: value}
    return {default_key: value}


def _normalize_structured_steps(steps: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for step in steps or []:
        if isinstance(step, dict):
            raw_type = step.get("reasoning_type") or step.get("step_type") or "analysis"
            step_type = CoTReasoningGenerator.STEP_TYPE_NORMALIZATION.get(raw_type, raw_type)
            description = step.get("description", "")
            steps_payload = step.get("steps") or []
            inputs = _ensure_dict(step.get("input") or step.get("inputs"), "value")
            outputs = _ensure_dict(step.get("output") or step.get("outputs"), "value")
            metadata = {
                key: value
                for key, value in step.items()
                if key not in {"reasoning_type", "step_type", "description", "steps", "input", "inputs", "output", "outputs"}
            }
            entry: Dict[str, Any] = {
                "reasoning_type": step_type,
            }
            if steps_payload:
                entry["steps"] = steps_payload
            elif description:
                entry["description"] = description
            if inputs:
                entry["inputs"] = inputs
            if outputs:
                entry["outputs"] = outputs
            if metadata:
                entry["metadata"] = metadata
        else:
            entry = {
                "reasoning_type": "analysis",
                "steps": [
                    {
                        "label": "step 1",
                        "description": str(step),
                    }
                ],
            }
        if not entry.get("inputs"):
            entry.pop("inputs", None)
        if not entry.get("outputs"):
            entry.pop("outputs", None)
        normalized.append(entry)
    return normalized


def _aggregate_reasoning_steps(
    steps: List[Any],
    cot_generator: CoTReasoningGenerator,
    answer_object: Optional[str],
) -> List[Dict[str, Any]]:
    aggregated: List[Dict[str, Any]] = []
    current_entry: Optional[Dict[str, Any]] = None

    def _append_new(reasoning_type: str) -> Dict[str, Any]:
        entry = {"reasoning_type": reasoning_type, "steps": []}
        aggregated.append(entry)
        return entry

    for raw in steps or []:
        if not isinstance(raw, dict):
            continue
        reasoning_type = raw.get("reasoning_type") or raw.get("step_type")
        if not reasoning_type:
            continue
        reasoning_type = cot_generator.STEP_TYPE_NORMALIZATION.get(reasoning_type, reasoning_type)

        if reasoning_type == "answer":
            description = raw.get("description")
            answer_entry: Dict[str, Any] = {"reasoning_type": "answer"}
            if description:
                answer_entry["description"] = cot_generator.clean_text(description)
            aggregated.append(answer_entry)
            current_entry = None
            continue

        sentences: List[str] = []
        if raw.get("steps"):
            for substep in raw["steps"]:
                if isinstance(substep, dict):
                    desc = substep.get("description")
                else:
                    desc = str(substep)
                if desc:
                    sentences.append(cot_generator.clean_text(desc))
        desc = raw.get("description")
        if desc:
            sentences.append(cot_generator.clean_text(desc))
        if reasoning_type == "elimination":
            eliminations = raw.get("inputs", {}).get("eliminations")
            if isinstance(eliminations, list):
                for elim in eliminations:
                    sentences.append(cot_generator.clean_text(str(elim)))
        if not sentences:
            continue

        if not aggregated or aggregated[-1].get("reasoning_type") != reasoning_type or aggregated[-1].get("reasoning_type") == "answer":
            current_entry = _append_new(reasoning_type)
        else:
            current_entry = aggregated[-1]

        for sentence in sentences:
            steps_list = current_entry.setdefault("steps", [])
            label = f"step {len(steps_list) + 1}"
            steps_list.append({"label": label, "description": sentence})

    return aggregated


def _build_reasoning_phases(
    question: Dict[str, Any],
    aggregated_steps: List[Dict[str, Any]],
    answer_object: Optional[str],
    object_list: List[str],
) -> List[Dict[str, Any]]:
    box_map = question.get("box_to_object") or {}
    phases: List[Dict[str, Any]] = []

    if box_map:
        phases.append(
            {
                "phase": "object_detection",
                "input": {"regions": list(box_map.keys())},
                "output": {"mappings": box_map},
                "steps": [f"{label} corresponds to {obj}" for label, obj in box_map.items()],
                "summary": "Scene objects identified.",
            }
        )

    phase_alias = {
        "question_analysis": "thinking",
        "attribute_analysis": "thinking",
        "concept_analysis": "thinking",
        "filter_analysis": "thinking",
        "spatial_analysis": "thinking",
        "spatial_calculation": "calculation",
        "elimination": "thinking",
        "thinking": "thinking",
    }

    last_thinking_summary: Optional[str] = None

    for entry in aggregated_steps:
        reasoning_type = entry.get("reasoning_type")
        if reasoning_type in {"answer", "object_recognition"}:
            continue

        phase = phase_alias.get(reasoning_type, "thinking")
        sentences: List[str] = [
            step.get("description")
            for step in entry.get("steps", [])
            if isinstance(step, dict) and step.get("description")
        ]
        sentences = [s for s in sentences if s]
        if not sentences:
            continue

        if phases and phases[-1].get("phase") == phase:
            phases[-1]["steps"].extend(sentences)
            phases[-1]["summary"] = sentences[-1]
        else:
            phases.append(
                {
                    "phase": phase,
                    "steps": sentences,
                    "summary": sentences[-1],
                }
            )

        if phase == "thinking":
            last_thinking_summary = sentences[-1]

    qtype = question.get("question_type", "")
    directional_candidates = {
        "spatial_left_right": ["left", "right"],
        "spatial_front_behind": ["front", "behind"],
        "spatial_above_below": ["above", "below"],
    }
    if qtype in directional_candidates:
        decision_candidates = directional_candidates[qtype]
        selected_decision = question.get("answer")
        if not selected_decision:
            selected_decision = answer_object
    else:
        decision_candidates = object_list
        selected_decision = answer_object

    if decision_candidates:
        selected_text = selected_decision if selected_decision is not None else "unknown"
        phases.append(
            {
                "phase": "decision",
                "input": {"candidates": decision_candidates},
                "output": {"selected_object": selected_decision},
                "summary": f"Selected {selected_text}.",
            }
        )

    final_answer = question.get("answer")
    rationale = last_thinking_summary or f"{answer_object} selected."
    phases.append(
        {
            "phase": "answer",
            "final_answer": final_answer,
            "rationale": rationale,
        }
    )

    return phases


def _extract_spatial_context(question: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    spatial_ctx: Dict[str, Any] = {}
    question_text = question.get("question") or ""
    if question_text:
        labels = re.findall(r"object (?:in|marked by)\s+(?:the\s+)?([A-Za-z ]+? box)", question_text, flags=re.IGNORECASE)
        if labels:
            spatial_ctx["target_label"] = labels[0].strip()
        if len(labels) > 1:
            spatial_ctx["reference_label"] = labels[1].strip()
    box_map = question.get("box_to_object") or {}
    target_label = spatial_ctx.get("target_label")
    reference_label = spatial_ctx.get("reference_label")
    if target_label and box_map.get(target_label):
        spatial_ctx["target_object_name"] = box_map[target_label]
    if reference_label and box_map.get(reference_label):
        spatial_ctx["reference_object_name"] = box_map[reference_label]
    target_phrase = _compose_spatial_phrase(target_label, spatial_ctx.get("target_object_name"))
    reference_phrase = _compose_spatial_phrase(reference_label, spatial_ctx.get("reference_object_name"))
    if target_phrase:
        spatial_ctx["target_phrase"] = target_phrase
    if reference_phrase:
        spatial_ctx["reference_phrase"] = reference_phrase
    answer_label = question.get("answer")
    if answer_label:
        spatial_ctx["answer_label"] = answer_label
    existing_context = question.get("spatial_context")
    if isinstance(existing_context, dict):
        for key, value in existing_context.items():
            spatial_ctx.setdefault(key, value)
    spatial_rel = meta.get("spatial_relationship")
    if isinstance(spatial_rel, dict):
        spatial_ctx["spatial_relationship"] = spatial_rel
    relation_map = {
        "spatial_left_right": "left_right",
        "spatial_front_behind": "front_behind",
        "spatial_above_below": "above_below",
        "spatial_closer_to_camera": "closer",
    }
    qtype = question.get("question_type", "")
    relation_key = relation_map.get(qtype)
    if relation_key and isinstance(spatial_ctx.get("spatial_relationship"), dict):
        relation_val = spatial_ctx["spatial_relationship"].get(relation_key)
        if relation_val:
            spatial_ctx["relation"] = relation_val
    return {key: value for key, value in spatial_ctx.items() if value}


def _build_spatial_details_from_metadata(
    spatial_context: Dict[str, Any],
    bbox_metadata: Dict[str, Any],
    pose_metadata: Dict[str, Any],
    question_type: str,
) -> Optional[Dict[str, Any]]:
    target_object = spatial_context.get("target_object_name")
    reference_object = spatial_context.get("reference_object_name")
    if not target_object or not reference_object:
        return None

    target_bbox_meta = bbox_metadata.get(target_object) or {}
    reference_bbox_meta = bbox_metadata.get(reference_object) or {}

    target_center = target_bbox_meta.get("center")
    reference_center = reference_bbox_meta.get("center")

    if (not target_center or not reference_center) and pose_metadata:
        target_pose = pose_metadata.get(target_object) or {}
        reference_pose = pose_metadata.get(reference_object) or {}
        target_center = target_center or target_pose.get("bbox_center")
        reference_center = reference_center or reference_pose.get("bbox_center")

    if not target_center or not reference_center:
        return None

    target_pose = pose_metadata.get(target_object) if pose_metadata else {}
    reference_pose = pose_metadata.get(reference_object) if pose_metadata else {}

    target_yaw = target_pose.get("yaw_deg")
    reference_yaw = reference_pose.get("yaw_deg")

    dx = float(target_center[0]) - float(reference_center[0])
    dy = float(target_center[1]) - float(reference_center[1])
    distance = math.hypot(dx, dy)

    relation_value = "unknown"
    axis_description = ""

    if question_type == "spatial_left_right":
        relation_value = "left" if dx < 0 else "right"
        if math.isclose(dx, 0.0, abs_tol=1e-3):
            axis_description = "The horizontal centers are aligned."
        else:
            direction = "right" if dx > 0 else "left"
            axis_description = f"The horizontal center offset is {abs(dx):.1f}px toward the {direction}."
    elif question_type == "spatial_above_below":
        relation_value = "above" if dy < 0 else "below"
        if math.isclose(dy, 0.0, abs_tol=1e-3):
            axis_description = "The vertical centers are aligned."
        else:
            direction = "below" if dy > 0 else "above"
            axis_description = f"The vertical center offset is {abs(dy):.1f}px toward the {direction}."
    elif question_type == "spatial_front_behind":
        relation_value = "front" if dy < 0 else "behind"
        if math.isclose(dy, 0.0, abs_tol=1e-3):
            axis_description = "The depth proxy offset is negligible."
        else:
            orientation = "front" if relation_value == "front" else "behind"
            axis_description = f"The depth proxy offset is {abs(dy):.1f}px, indicating {orientation}."
    elif question_type == "spatial_closer_to_camera":
        relation_value = "target" if dy < 0 else "reference"
        if math.isclose(dy, 0.0, abs_tol=1e-3):
            axis_description = "Vertical ordering suggests both objects are at similar depth."
        else:
            closer_hint = (
                "the target appears lower in the frame (closer to camera)"
                if dy < 0
                else "the reference appears lower in the frame (closer to camera)"
            )
            axis_description = f"The vertical ordering offset is {abs(dy):.1f}px, so {closer_hint}."

    spatial_relationship = {
        "left_right": relation_value if question_type == "spatial_left_right" else None,
        "above_below": relation_value if question_type == "spatial_above_below" else None,
        "front_behind": relation_value if question_type == "spatial_front_behind" else None,
        "closer": relation_value if question_type == "spatial_closer_to_camera" else None,
    }

    return {
        "target": {
            "object": target_object,
            "label": spatial_context.get("target_label"),
            "center": target_center,
            "bbox": target_bbox_meta.get("bbox"),
            "yaw_deg": target_yaw,
        },
        "reference": {
            "object": reference_object,
            "label": spatial_context.get("reference_label"),
            "center": reference_center,
            "bbox": reference_bbox_meta.get("bbox"),
            "yaw_deg": reference_yaw,
        },
        "delta": {
            "dx": dx,
            "dy": dy,
            "distance": distance,
            "axis_description": axis_description,
        },
        "spatial_relationship": spatial_relationship,
        "relation_value": relation_value,
    }


def _determine_filter_category(question_type: str) -> Tuple[Optional[str], Optional[str]]:
    if not question_type:
        return None, None
    if question_type.startswith("function_"):
        return "Function", "final_taxonomy_function"
    if question_type.startswith("affordance_"):
        return "Affordance", "final_taxonomy_affordances"
    if question_type == "physical_property":
        return "Physical Property", "final_taxonomy_physical_properties"
    if question_type == "material_property" or question_type.startswith("material_"):
        return "Material", "final_taxonomy_material"
    return None, None


def _get_taxonomy_clusters(
    taxonomy_utils: TaxonomyUtils,
    object_name: str,
    taxonomy_key: str,
) -> List[str]:
    if not object_name or not taxonomy_key:
        return []
    try:
        clusters = taxonomy_utils.get_object_clusters(object_name, taxonomy_key)
        if not clusters:
            return []
        return [cluster for cluster in clusters if cluster]
    except Exception:
        return []


def _build_filter_chain(
    question_type: str,
    answer_object: str,
    object_list: List[str],
    taxonomy_utils: TaxonomyUtils,
) -> Optional[List[Dict[str, Any]]]:
    filter_label, taxonomy_key = _determine_filter_category(question_type)
    if not filter_label or not taxonomy_key or not taxonomy_utils:
        return None

    answer_clusters = _get_taxonomy_clusters(taxonomy_utils, answer_object, taxonomy_key)
    if not answer_clusters:
        return None

    object_clusters: Dict[str, List[str]] = {
        obj: _get_taxonomy_clusters(taxonomy_utils, obj, taxonomy_key)
        for obj in object_list
    }

    objects_before = list(object_list)
    selected_cluster = None
    objects_after: List[str] = []

    for cluster in answer_clusters:
        matching = [obj for obj, clusters in object_clusters.items() if cluster in clusters]
        if answer_object not in matching:
            continue
        selected_cluster = cluster
        objects_after = matching
        # Prefer clusters that uniquely identify the answer
        if len(matching) == 1:
            break

    if not selected_cluster or not objects_after:
        return None

    return [
        {
            "filter_type": filter_label,
            "filter_value": selected_cluster,
            "objects_before": objects_before,
            "objects_after": objects_after,
        }
    ]


def _gather_reasoning_kwargs(
    question: Dict[str, Any],
    answer_object: Optional[str],
    object_list: List[str],
    object_utils: ObjectUtils,
    *,
    annotations_dir: Optional[Path] = None,
    image_id: Optional[str] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    meta = question.get("question_metadata") or {}
    qtype = question.get("question_type", "")

    if not answer_object:
        return kwargs

    if qtype.startswith("spatial_"):
        spatial_context = _extract_spatial_context(question, meta)
        if spatial_context:
            scene_lookup_id = (
                question.get("source_scene_path")
                or question.get("source_scene_id")
                or image_id
            )
            if scene_lookup_id:
                spatial_context["source_scene_path"] = scene_lookup_id
            bbox_metadata = question.get("bbox_metadata") or {}
            pose_metadata = question.get("pose_metadata") or {}
            spatial_details = None
            if annotations_dir and scene_lookup_id:
                spatial_details = object_utils.get_spatial_reasoning_details(
                    image_id=scene_lookup_id,
                    question_type=qtype,
                    target_object=spatial_context.get("target_object_name") or answer_object,
                    reference_object=spatial_context.get("reference_object_name"),
                    annotations_dir=annotations_dir,
                    target_label=spatial_context.get("target_label"),
                    reference_label=spatial_context.get("reference_label"),
                )
            if not spatial_details and bbox_metadata:
                spatial_details = _build_spatial_details_from_metadata(
                    spatial_context,
                    bbox_metadata,
                    pose_metadata,
                    qtype,
                )
            if spatial_details:
                spatial_context["calculation_details"] = spatial_details
                relation_value = spatial_details.get("relation_value")
                if relation_value and "relation" not in spatial_context:
                    spatial_context["relation"] = relation_value
                spatial_rel = spatial_details.get("spatial_relationship")
                if spatial_rel and "spatial_relationship" not in spatial_context:
                    spatial_context["spatial_relationship"] = spatial_rel
            kwargs["spatial_context"] = spatial_context

    if qtype == "material_property":
        material = meta.get("material") or object_utils.get_object_material(answer_object)
        if material:
            kwargs["material"] = material
    elif qtype == "function_knowledge":
        function = meta.get("function") or object_utils.get_object_function(answer_object)
        if function:
            kwargs["function"] = function
    if qtype == "physical_property":
        physical_property = meta.get("physical_property")
        if not physical_property:
            props = object_utils.get_object_physical_properties(answer_object)
            if props:
                physical_property = props[0]
        if physical_property:
            kwargs["physical_property"] = physical_property
    if qtype.startswith("description_"):
        description = meta.get("description") or object_utils.get_object_description(answer_object)
        if description:
            kwargs["description"] = description

    filter_chain = question.get("filter_chain")
    if filter_chain:
        kwargs["filter_chain"] = filter_chain

    return kwargs


def refresh_reasoning(
    questions: List[Dict[str, Any]],
    taxonomy_utils: TaxonomyUtils,
    object_utils: ObjectUtils,
    annotations_dir: Optional[Path] = None,
    reasoning_output: Optional[Path] = None,
) -> int:
    cot_generator = CoTReasoningGenerator(taxonomy_utils=taxonomy_utils)
    refreshed = 0
    reasoning_records: List[Dict[str, Any]] = []

    for question in questions:
        qtype = question.get("question_type", "")
        answer_object = _resolve_answer_object(question)
        object_list = _resolve_object_list(question)
        if not object_list and question.get("objects"):
            object_list = [obj for obj in question["objects"] if obj]
        if answer_object is None:
            LOGGER.debug("Skipping question_index=%s (unable to resolve answer object)", question.get("question_index"))
            continue
        if not object_list:
            LOGGER.debug("Skipping question_index=%s (no object list)", question.get("question_index"))
            continue

        if not question.get("filter_chain"):
            filter_chain = _build_filter_chain(
                qtype,
                answer_object,
                object_list,
                taxonomy_utils,
            )
            if filter_chain:
                question["filter_chain"] = filter_chain

        kwargs = _gather_reasoning_kwargs(
            question,
            answer_object,
            object_list,
            object_utils,
            annotations_dir=annotations_dir,
            image_id=question.get("image_id"),
        )

        try:
            filter_chain_arg = kwargs.pop("filter_chain", None)
            box_to_object = question.get("box_to_object")
            raw_reasoning = cot_generator.generate_comprehensive_reasoning(
                qtype,
                answer_object,
                object_list,
                answer_object,
                filter_chain=filter_chain_arg,
                box_to_object=box_to_object,
                **kwargs,
            )
            aggregated_reasoning = _aggregate_reasoning_steps(
                raw_reasoning,
                cot_generator,
                answer_object,
            )
            structured_reasoning = _build_reasoning_phases(
                question,
                aggregated_reasoning,
                answer_object,
                object_list,
            )
            if not isinstance(structured_reasoning, list):
                LOGGER.warning(
                    "Reasoning generator returned unexpected type (%s) for question_index=%s; skipping update.",
                    type(structured_reasoning),
                    question.get("question_index"),
                )
                continue
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning(
                "Failed to regenerate reasoning for question_index=%s (%s): %s",
                question.get("question_index"),
                qtype,
                exc,
            )
            continue

        rendered_reasoning = cot_generator.render_reasoning_text(structured_reasoning)

        question_index = question.get("question_index")

        if reasoning_output:
            question.pop("reasoning", None)
            question.pop("reasoning_text", None)
            reasoning_records.append(
                {
                    "question_index": question_index,
                    "question_type": qtype,
                    "reasoning": structured_reasoning,
                    "reasoning_text": rendered_reasoning,
                }
            )
            refreshed += 1
        else:
            existing_structured = question.get("reasoning")
            existing_rendered = question.get("reasoning_text") or (
                existing_structured if isinstance(existing_structured, str) else None
            )
            structured_changed = existing_structured != structured_reasoning
            rendered_changed = rendered_reasoning and rendered_reasoning != existing_rendered

            if structured_changed:
                question["reasoning"] = structured_reasoning
            if rendered_reasoning:
                question["reasoning_text"] = rendered_reasoning

            if structured_changed or rendered_changed:
                refreshed += 1

    if reasoning_output:
        reasoning_output.parent.mkdir(parents=True, exist_ok=True)
        with reasoning_output.open("w", encoding="utf-8") as handle:
            json.dump(reasoning_records, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    return refreshed


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_questions(path: Path, questions: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(questions, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.output and args.in_place:
        raise ValueError("Specify either --output or --in-place, not both.")
    if not args.output and not args.in_place:
        raise ValueError("Provide --output path or use --in-place to overwrite the input file.")

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    LOGGER.info("Loading questions from %s", input_path)
    questions = load_questions(input_path)

    taxonomy_utils = TaxonomyUtils(args.taxonomy_dir.resolve())
    object_utils = ObjectUtils(taxonomy_utils=taxonomy_utils)

    annotations_dir: Optional[Path] = None
    if args.annotations_dir:
        annotations_dir = args.annotations_dir.resolve()
    else:
        meta_path = input_path.parent / "generation_metadata.json"
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
                candidate = metadata.get("images_dir")
                if candidate:
                    candidate_path = Path(candidate)
                    if candidate_path.exists():
                        annotations_dir = candidate_path
                        LOGGER.info("Inferred annotations directory from generation metadata: %s", annotations_dir)
            except Exception as meta_error:
                LOGGER.debug("Failed to parse generation metadata for annotations dir: %s", meta_error)
        if annotations_dir is None:
            default_real = Path("/path/to/SpatialReasonerDataGen/qa_gen/openimages_unified_output")
            if default_real.exists():
                annotations_dir = default_real
                LOGGER.info("Falling back to default real-image annotations directory: %s", annotations_dir)
    if annotations_dir and not annotations_dir.exists():
        LOGGER.warning("Annotations directory %s does not exist; spatial reasoning details may be limited.", annotations_dir)
        annotations_dir = None
    elif annotations_dir:
        LOGGER.info("Using annotations directory: %s", annotations_dir)
    else:
        LOGGER.warning("No annotations directory provided; spatial reasoning will omit numeric details.")

    reasoning_output_path: Optional[Path] = args.reasoning_output.resolve() if args.reasoning_output else None
    if args.in_place and reasoning_output_path is None:
        reasoning_output_path = input_path.with_name(f"{input_path.stem}_reasoning.json")

    LOGGER.info("Regenerating reasoning strings using current templates...")
    updated_count = refresh_reasoning(
        questions,
        taxonomy_utils,
        object_utils,
        annotations_dir=annotations_dir,
        reasoning_output=reasoning_output_path,
    )
    LOGGER.info("Updated reasoning for %d question(s).", updated_count)

    LOGGER.info("Writing output to %s", output_path)
    write_questions(output_path, questions)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
