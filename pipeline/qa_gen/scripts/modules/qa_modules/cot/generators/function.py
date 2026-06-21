"""
Function reasoning generator.
"""

from __future__ import annotations

import string
from typing import Any, Dict, List


def generate_function_reasoning(
    reasoner: "CoTReasoningGenerator",
    question_type: str,
    target_object: str,
    object_set: List[str],
    answer: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Generate structured function reasoning for any function type."""
    del kwargs
    if question_type in ("function_seating", "functional_seating"):
        return reasoner.generate_affordance_reasoning(
            "affordance_sit__ride__attend", target_object, object_set, answer
        )

    function_key = question_type.replace("function_", "").replace("_", "_")

    template = reasoner._find_template_match(function_key, reasoner.function_templates)

    if template:
        objects_str = ", ".join(object_set)
        template_placeholders = [
            field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name
        ]
        format_kwargs = {k: v for k, v in {"objects": objects_str, "answer": answer}.items() if k in template_placeholders}
        analysis_sentence = reasoner.clean_text(template.format(**format_kwargs))
        steps = [
            reasoner._create_reasoning_step("functional_analysis", analysis_sentence),
            reasoner._create_reasoning_step(
                "thinking",
                f"The {answer} fulfills the required function while other objects do not",
            ),
        ]
    else:
        steps = reasoner._generate_taxonomy_enhanced_reasoning(
            question_type, target_object, object_set, answer, "function"
        )

    return steps

