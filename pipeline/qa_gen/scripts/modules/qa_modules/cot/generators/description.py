"""
Description-matching reasoning generator.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..core import format_object_list


def generate_description_reasoning(
    reasoner: "CoTReasoningGenerator",
    question_type: str,
    target_object: str,
    object_set: List[str],
    answer: str,
    description: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Generate reasoning for description questions using a concise statement."""
    del kwargs  # Compatibility with previous signature.
    description_clean = reasoner.clean_text(description) if description else "the description"
    others = [obj for obj in object_set if obj != answer]
    if not others:
        suffix = "and no other objects are present for comparison."
    else:
        others_phrase = format_object_list(others)
        aux_do = "does" if len(others) == 1 else "do"
        suffix = f"while {others_phrase} {aux_do} not."

    if "non" in question_type.lower():
        sentence = (
            f"The {answer} does not match the description {description_clean}, "
            f"{suffix}"
        )
    else:
        sentence = (
            f"The {answer} matches the description {description_clean} "
            f"{suffix}"
        )

    return [
        reasoner._create_reasoning_step(
            "thinking",
            sentence,
            output_data={"answer": answer},
        )
    ]

