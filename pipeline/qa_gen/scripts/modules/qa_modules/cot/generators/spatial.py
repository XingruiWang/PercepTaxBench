"""
Spatial reasoning generator functions.

These helpers operate on a CoTReasoningGenerator instance but live outside the
class to keep domain-specific logic modular.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def build_spatial_relation_clause(
    reasoner: "CoTReasoningGenerator",
    question_type: str,
    relation_value: Optional[str],
    target_phrase: str,
    reference_phrase: str,
    answer_label: Optional[str] = None,
) -> str:
    target_phrase = target_phrase or "the compared object"
    reference_phrase = reference_phrase or "the reference object"
    relation_value = (relation_value or "").lower()
    answer_term = (answer_label or "").lower()

    if question_type == "spatial_left_right":
        if relation_value in {"left", "right"}:
            clause = f"{target_phrase} sits on the {relation_value} side of {reference_phrase}."
        elif answer_term in {"left", "right"}:
            clause = f"{target_phrase} sits on the {answer_term} side of {reference_phrase}."
        else:
            clause = (
                f"{target_phrase} occupies the side of {reference_phrase} described in the question, "
                "while the other option falls on the opposite side."
            )
    elif question_type == "spatial_front_behind":
        if relation_value == "front":
            clause = f"{target_phrase} stands in front of {reference_phrase} along that facing direction."
        elif relation_value == "behind":
            clause = f"{target_phrase} sits behind {reference_phrase} once the front is established."
        elif answer_term in {"front", "behind"}:
            clause = f"{target_phrase} aligns on the {answer_term} side of {reference_phrase}, unlike the other option."
        else:
            clause = (
                f"{target_phrase} aligns with the front-or-behind relationship described for {reference_phrase}, "
                "unlike the alternatives."
            )
    elif question_type == "spatial_above_below":
        if relation_value in {"above", "below"}:
            clause = f"{target_phrase} is {relation_value} {reference_phrase}, matching the vertical arrangement."
        elif answer_term in {"above", "below"}:
            clause = f"{target_phrase} is {answer_term} {reference_phrase}, matching the vertical arrangement."
        else:
            clause = (
                f"{target_phrase} occupies the correct vertical position relative to {reference_phrase}, "
                "whereas others do not."
            )
    elif question_type == "spatial_closer_to_camera":
        if relation_value == "closer":
            clause = (
                f"{target_phrase} appears closer to the camera than {reference_phrase} based on scale and occlusion cues."
            )
        elif relation_value == "farther":
            clause = (
                f"{target_phrase} appears farther from the camera than {reference_phrase} when perspective cues are evaluated."
            )
        elif answer_term in {"closer", "farther"}:
            clause = (
                f"{target_phrase} appears {answer_term} relative to {reference_phrase} when perspective cues are evaluated."
            )
        else:
            clause = f"{target_phrase} matches the depth relationship described relative to {reference_phrase}."
    else:
        clause = f"{target_phrase} matches the spatial relationship to {reference_phrase} described in the question."

    if clause and clause[0].islower():
        clause = clause[0].upper() + clause[1:]
    return clause


def _format_orientation_line(phrase: str, info: Dict[str, Any]) -> str:
    descriptor = phrase[0].upper() + phrase[1:] if phrase else "The object"
    center = info.get("center")
    yaw = info.get("yaw_deg")
    parts: List[str] = []
    if center:
        parts.append(f"center at x {center[0]:.1f}px, y {center[1]:.1f}px")
    if yaw is not None:
        parts.append(f"yaw {yaw:.1f} degrees")
    if not parts:
        return descriptor
    return f"{descriptor} has " + " and ".join(parts)


def _compose_axis_description(
    reasoner: "CoTReasoningGenerator",
    question_type: str,
    delta_meta: Dict[str, Any],
    relation: Optional[str],
    default_text: Optional[str],
) -> Optional[str]:
    if not delta_meta:
        return default_text
    relation_normalized = (relation or "").lower()
    dx = delta_meta.get("dx")
    dy = delta_meta.get("dy")

    def _format_offset(
        value: Optional[float],
        axis_word: str,
        relation_word: Optional[str],
        fallback_direction: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        if math.isclose(value, 0.0, abs_tol=1e-3):
            return f"The {axis_word} centers are aligned."
        if relation_word:
            return (
                f"The {axis_word} center offset is {abs(value):.1f}px, so the compared object lies to the {relation_word} side."
                if axis_word == "horizontal"
                else f"The {axis_word} center offset is {abs(value):.1f}px, placing the compared object {relation_word} the reference."
            )
        direction_word = fallback_direction or ("positive" if value > 0 else "negative")
        return f"The {axis_word} center offset is {abs(value):.1f}px toward the {direction_word}."

    if question_type == "spatial_left_right":
        fallback = "right" if (dx or 0) > 0 else "left"
        relation_word = relation_normalized if relation_normalized in {"left", "right"} else None
        return _format_offset(dx, "horizontal", relation_word, fallback) or default_text
    if question_type == "spatial_above_below":
        fallback = "below" if (dy or 0) > 0 else "above"
        relation_word = relation_normalized if relation_normalized in {"above", "below"} else None
        return _format_offset(dy, "vertical", relation_word, fallback) or default_text
    if question_type == "spatial_front_behind":
        if dy is None:
            return default_text
        if math.isclose(dy, 0.0, abs_tol=1e-3):
            return "The depth proxy offset is negligible."
        orientation = relation_normalized if relation_normalized in {"front", "behind"} else ("front" if dy < 0 else "behind")
        return f"The depth proxy offset is {abs(dy):.1f}px, indicating {orientation}."
    if question_type == "spatial_closer_to_camera":
        if dy is None:
            return default_text
        if math.isclose(dy, 0.0, abs_tol=1e-3):
            return "Vertical ordering suggests both objects are at similar depth."
        if relation_normalized == "closer":
            hint = "the compared object appears lower in the frame (closer to camera)"
        elif relation_normalized == "farther":
            hint = "the compared object appears higher in the frame (farther from camera)"
        else:
            hint = (
                "the compared object appears lower in the frame (closer to camera)"
                if dy < 0
                else "the reference appears lower in the frame (closer to camera)"
            )
        return f"The vertical ordering offset is {abs(dy):.1f}px, so {hint}."
    return default_text


def generate_spatial_reasoning(
    reasoner: "CoTReasoningGenerator",
    question_type: str,
    target_object: str,
    object_set: List[str],
    answer: str,
    spatial_context: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Generate structured spatial reasoning for spatial question types with orientation emphasis."""
    del kwargs  # Unused but kept for parity with class signature.
    reasoner._last_spatial_answer_label = None

    reference_phrase = "the reference object"
    target_phrase = "the compared object"
    answer_label = answer
    relation_key_map = {
        "spatial_left_right": "left_right",
        "spatial_front_behind": "front_behind",
        "spatial_above_below": "above_below",
        "spatial_closer_to_camera": "closer",
    }
    relation_value = None
    calculation_details = None
    if spatial_context:
        reference_phrase = spatial_context.get("reference_phrase", reference_phrase)
        target_phrase = spatial_context.get("target_phrase", target_phrase)
        answer_label = spatial_context.get("answer_label", answer_label)
        if isinstance(spatial_context.get("relation"), str):
            relation_value = spatial_context["relation"]
        spatial_rel = spatial_context.get("spatial_relationship")
        relation_key = relation_key_map.get(question_type)
        if not relation_value and relation_key and isinstance(spatial_rel, dict):
            relation_value = spatial_rel.get(relation_key)
        calculation_details = spatial_context.get("calculation_details")

    relation_aliases = {
        "spatial_left_right": {"left", "right"},
        "spatial_front_behind": {"front", "behind"},
        "spatial_above_below": {"above", "below"},
        "spatial_closer_to_camera": {"closer", "farther", "target", "reference"},
    }
    allowed_terms = relation_aliases.get(question_type, set())
    canonical_relation = (relation_value or "").lower()
    normalized_answer = (answer_label or "").lower()
    if question_type == "spatial_closer_to_camera":
        if canonical_relation == "target":
            canonical_relation = "closer"
        elif canonical_relation == "reference":
            canonical_relation = "farther"
    if canonical_relation not in allowed_terms and normalized_answer in allowed_terms:
        canonical_relation = normalized_answer
    if normalized_answer not in allowed_terms and canonical_relation in allowed_terms:
        answer_label = canonical_relation
    relation_value = canonical_relation if canonical_relation else None

    axis_names = {
        "spatial_left_right": "horizontal (x)",
        "spatial_front_behind": "depth (z)",
        "spatial_above_below": "vertical (y)",
        "spatial_closer_to_camera": "depth (z)",
    }
    axis_name = axis_names.get(question_type, "relevant")

    steps: List[Dict[str, Any]] = []

    if calculation_details:
        reference_info = calculation_details.get("reference", {}) or {}
        target_info = calculation_details.get("target", {}) or {}
        delta_info = calculation_details.get("delta", {}) or {}
        relation_value = calculation_details.get("relation_value", relation_value)
        if isinstance(relation_value, str):
            relation_value = relation_value.lower()
            if question_type == "spatial_closer_to_camera":
                if relation_value == "target":
                    relation_value = "closer"
                elif relation_value == "reference":
                    relation_value = "farther"
        relation_clause = build_spatial_relation_clause(
            reasoner,
            question_type,
            relation_value,
            target_phrase,
            reference_phrase,
            answer_label=answer_label,
        )

        if reference_info:
            steps.append(
                reasoner._create_reasoning_step(
                    "orientation",
                    _format_orientation_line(reference_phrase, reference_info),
                    input_data=reference_info,
                )
            )
        else:
            steps.append(
                reasoner._create_reasoning_step(
                    "orientation",
                    f"I anchor on {reference_phrase} to establish the scene orientation.",
                )
            )

        if target_info:
            steps.append(
                reasoner._create_reasoning_step(
                    "orientation",
                    _format_orientation_line(target_phrase, target_info),
                    input_data=target_info,
                )
            )
        else:
            steps.append(
                reasoner._create_reasoning_step(
                    "orientation",
                    f"I locate {target_phrase} relative to {reference_phrase} along the {axis_name} axis.",
                )
            )

        axis_description = delta_info.get("axis_description")
        axis_description = _compose_axis_description(
            reasoner, question_type, delta_info, relation_value, axis_description
        )
        if axis_description and relation_clause:
            calc_sections = [
                axis_description.rstrip(".").rstrip(),
                relation_clause.rstrip(".").rstrip(),
            ]
            calculation_text = ". ".join(section for section in calc_sections if section) + "."
        elif axis_description:
            calculation_text = axis_description.rstrip(".") + "."
        else:
            base_text = relation_clause or (
                f"I compare bounding box centers along the {axis_name} axis "
                f"to evaluate the relation between {target_phrase} and {reference_phrase}."
            )
            calculation_text = base_text.rstrip(".") + "."
        steps.append(
            reasoner._create_reasoning_step(
                "calculation",
                calculation_text,
                input_data=delta_info,
                output_data={"relation_value": relation_value or "unknown"},
            )
        )

        if relation_value and relation_value not in {"unknown", None}:
            if relation_value == "target":
                relation_phrase = f"closer to the camera than {reference_phrase}"
            elif relation_value == "reference":
                relation_phrase = f"farther from the camera than {reference_phrase}"
            else:
                relation_phrase = f"{relation_value} of {reference_phrase}"
            reasoning_summary = (
                f"These measurements show that {target_phrase} is {relation_phrase}, so {answer_label} is correct."
            )
        else:
            reasoning_summary = (
                f"These cues confirm that {answer_label} satisfies the spatial relationship while alternatives do not."
            )
        steps.append(
            reasoner._create_reasoning_step(
                "thinking",
                reasoning_summary,
                output_data={"answer": answer_label, "relation": relation_value or "unknown"},
            )
        )
    else:
        relation_clause = build_spatial_relation_clause(
            reasoner,
            question_type,
            relation_value,
            target_phrase,
            reference_phrase,
            answer_label=answer_label,
        )
        steps.append(
            reasoner._create_reasoning_step(
                "orientation",
                f"I anchor on {reference_phrase} to establish the scene orientation.",
            )
        )
        steps.append(
            reasoner._create_reasoning_step(
                "orientation",
                f"I locate {target_phrase} relative to {reference_phrase} along the {axis_name} axis.",
            )
        )

        steps.append(
            reasoner._create_reasoning_step(
                "calculation",
                relation_clause
                and f"I compare bounding box centers along the {axis_name} axis: {relation_clause}"
                or f"I compare bounding box centers along the {axis_name} axis to evaluate the relation between {target_phrase} and {reference_phrase}.",
            )
        )

        steps.append(
            reasoner._create_reasoning_step(
                "thinking",
                f"These cues confirm that {answer_label} satisfies the spatial relationship while alternatives do not.",
                output_data={"answer": answer_label},
            )
        )
    if relation_value in {"left", "right", "front", "behind", "above", "below", "closer", "farther"}:
        reasoner._last_spatial_answer_label = relation_value
    else:
        reasoner._last_spatial_answer_label = answer_label if answer_label else None

    return steps

