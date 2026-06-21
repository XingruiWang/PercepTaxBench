#!/usr/bin/env python3
"""
Heuristics for inferring legacy question_type values from question text.

Older benchmarks (e.g., taxonomyQABench_realimage_v2) stored only high level
question_category labels. When those questions are merged into newer
benchmarks, the detailed question_type field can be missing. This module
provides lightweight string-matching utilities that recognise the legacy
phrasing and recover the most appropriate detailed type so downstream
pipelines stay consistent.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


def _base_question_text(question: Dict) -> str:
    """Extract the lead clause of the question for pattern matching."""
    text = (question.get("original_question") or question.get("question") or "").lower()
    if not text:
        return ""

    # Remove trailing metadata (option listings, clauses after '?', etc.)
    for delimiter in (
        "? objects to choose from",
        "? option objects",
        " objects to choose from",
        " option objects",
        " objects:",
    ):
        if delimiter in text:
            text = text.split(delimiter, 1)[0]
    if "?" in text:
        text = text.split("?", 1)[0]
    return text.strip()


_REPURPOSING_MAP = {
    "reflector": "repurposing_reflector_concept",
    "container": "repurposing_container_concept",
    "cushion": "repurposing_cushion_concept",
    "shield": "repurposing_shield_concept",
    "stepstool": "repurposing_stepstool_concept",
    "bookend": "repurposing_bookend_concept",
    "lever": "repurposing_lever_concept",
}


_AFFORDANCE_SUFFIX_MAP = {
    "wearables and apparel": "affordance_wearables_and_apparel",
    "sit ride attend": "affordance_sit__ride__attend",
    "build span occupy": "affordance_build__span__occupy",
    "tableware and serveware": "affordance_tableware_and_serveware",
    "interact with living moving things": "affordance_interact_with_living_moving_things",
    "display exhibit signal value": "affordance_display__exhibit__signal_value",
    "mechanical control": "affordance_mechanical_control",
    "enclosures and venues enter use": "affordance_enclosures_and_venues_(enter_use)",
    "place support work on": "affordance_place__support__work_on",
    "grip carry operate": "affordance_grip__carry__operate",
    "operate use device": "affordance_operate__use_device",
    "grow plant vegetation": "affordance_grow__plant_(vegetation)",
    "tableware_and_serveware": "affordance_tableware_and_serveware",
    "architectural components and fixtures": "affordance_architectural_components_and_fixtures",
    "art display view appraise": "affordance_art_display_(view_appraise)",
    "structured operational engagement": "affordance_structured_operational_engagement",
    "household facility operations": "affordance_household__facility_operations",
}


_SIMPLE_SUBSTRING_MAP = {
    "which object would be most affected by high heat": "counterfactual_heat",
    "if water spills, which object gets damaged first": "counterfactual_water",
    "which object is rigid, movable, but not designed as a container": "compositional_set_subtraction_container",
    "which object is hollow": "compositional_set_subtraction_hollow",
    "which object can hide small items while keeping the area tidy": "latent_containment",
    "which object can be compressed to fit in tight spaces": "latent_compressible",
    "which object can be folded or collapsed to save space": "functional_foldable",
    "which object can be used for sitting or resting": "functional_seating",
    "which object is furniture": "affordance_furniture",
    "which object can have items placed on it": "affordance_place__support__work_on",
    "which object can contain or carry items": "affordance_contain__carry__package",
    "which object can be gripped and carried": "affordance_grip__carry__operate",
    "which object can be operated or used": "affordance_operate__use_device",
    "which object is used for growing plants": "affordance_grow__plant_(vegetation)",
    "which object is an enclosed space or venue": "affordance_enclosures_and_venues_(enter_use)",
    "which object is prepared food": "affordance_food_—_prepared_dishes",
    "which object is food or produce": "affordance_food_—_ingredients_and_produce",
    "which object involves reading or communication": "affordance_mediated_action_and_meaning",
    "which object controls or produces light": "affordance_control__express__light",
    "which object could be repurposed as a cushion": "repurposing_cushion_concept",
    "which object could be repurposed as a shield": "repurposing_shield_concept",
    "which object could be repurposed as a stepstool": "repurposing_stepstool_concept",
    "which object could be repurposed as a container": "repurposing_container_concept",
    "which object could be repurposed as a reflector": "repurposing_reflector_concept",
    "which object could be repurposed as a bookend": "repurposing_bookend_concept",
    "which object could be repurposed as a lever": "repurposing_lever_concept",
    "which object has the affordance of sit ride attend": "affordance_sit__ride__attend",
    "which object has the affordance of wearables and apparel": "affordance_wearables_and_apparel",
    "which object has the affordance of tableware and serveware": "affordance_tableware_and_serveware",
    "which object has the affordance of build span occupy": "affordance_build__span__occupy",
    "which object has the affordance of interact with living moving things": "affordance_interact_with_living_moving_things",
    "which object has the affordance of display exhibit signal value": "affordance_display__exhibit__signal_value",
}


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("—", " ")
    text = text.replace("–", " ")
    text = text.replace("‑", " ")
    text = text.replace("&", " and ")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def infer_legacy_question_type(question: Dict) -> Optional[str]:
    """
    Attempt to infer a detailed question_type from legacy question text.

    Returns:
        The inferred detailed question_type string, or None if no match is found.
    """
    lead_text = _base_question_text(question)
    if not lead_text:
        return None

    # Direct substring checks for well-known phrasing.
    for substring, q_type in _SIMPLE_SUBSTRING_MAP.items():
        if substring in lead_text:
            return q_type

    # Repurposing variants.
    if "repurposed as a" in lead_text:
        for noun, q_type in _REPURPOSING_MAP.items():
            if f"repurposed as a {noun}" in lead_text:
                return q_type

    # Affordance phrasing ("has the affordance of ...").
    if "has the affordance of" in lead_text:
        suffix = lead_text.split("has the affordance of", 1)[1]
        normalized = _normalize_text(suffix)
        if normalized in _AFFORDANCE_SUFFIX_MAP:
            return _AFFORDANCE_SUFFIX_MAP[normalized]

    # Material / description / function patterns.
    if "matches this description" in lead_text:
        return "description_matching"
    if "is made of" in lead_text or "is made from" in lead_text:
        return "material_property"
    if "is used as" in lead_text or "is used for" in lead_text:
        return "function_knowledge"

    # Spatial relations.
    if "above or below" in lead_text:
        return "spatial_above_below"
    if "left or right" in lead_text:
        return "spatial_left_right"
    if "in front of or behind" in lead_text:
        return "spatial_front_behind"
    if "closer to the camera" in lead_text:
        return "spatial_closer_to_camera"

    return None


__all__ = ["infer_legacy_question_type"]


