"""
Affordance reasoning generator.
"""

from __future__ import annotations

from typing import Any, Dict, List


def generate_affordance_reasoning(
    reasoner: "CoTReasoningGenerator",
    question_type: str,
    target_object: str,
    object_set: List[str],
    answer: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Generate structured affordance reasoning for any affordance type."""
    del (target_object, object_set, kwargs)
    affordance_key = question_type.replace("affordance_", "")

    normalized_key = affordance_key.replace("__", "_").replace("_", "_")

    template_data = None

    if normalized_key in reasoner.affordance_templates:
        template_data = reasoner.affordance_templates[normalized_key]

    if not template_data:
        for key, data in reasoner.affordance_templates.items():
            clean_key = (
                key.replace("(", "")
                .replace(")", "")
                .replace("_", "")
                .replace("—", "")
                .replace(",", "")
                .replace(" ", "")
                .lower()
            )
            clean_affordance = (
                normalized_key.replace("(", "")
                .replace(")", "")
                .replace("_", "")
                .replace("—", "")
                .replace(",", "")
                .replace(" ", "")
                .lower()
            )

            if clean_affordance in clean_key or clean_key in clean_affordance:
                template_data = data
                break

    if template_data:
        features = ", ".join(template_data["features"][:2])
        requirements = ", ".join(template_data["requirements"][:2])
        steps = [
            reasoner._create_reasoning_step(
                "affordance_analysis",
                f"The {answer} offers features such as {features} and meets requirements like {requirements}",
            ),
            reasoner._create_reasoning_step(
                "thinking",
                f"These affordances make the {answer} suitable while alternatives fail to meet all requirements",
            ),
        ]
    else:
        steps = reasoner._generate_taxonomy_enhanced_reasoning(
            question_type, target_object, object_set, answer, "affordance"
        )

    return steps

