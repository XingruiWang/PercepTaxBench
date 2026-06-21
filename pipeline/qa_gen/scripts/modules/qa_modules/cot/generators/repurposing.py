"""
Repurposing reasoning generator.
"""

from __future__ import annotations

from typing import Any, Dict, List


def generate_repurposing_reasoning(
    reasoner: "CoTReasoningGenerator",
    question_type: str,
    target_object: str,
    object_set: List[str],
    answer: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Generate reasoning for repurposing questions using templates when available."""
    del (target_object, kwargs)
    repurposing_key = question_type.replace("repurposing_", "").replace("_concept", "")
    template = reasoner._find_template_match(repurposing_key, reasoner.repurposing_templates)

    if template:
        objects_str = ", ".join(object_set)
        format_kwargs = {"objects": objects_str, "answer": answer}
        import string

        template_placeholders = [
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name
        ]
        filtered_kwargs = {k: v for k, v in format_kwargs.items() if k in template_placeholders}
        analysis_sentence = reasoner.clean_text(template.format(**filtered_kwargs))
        qa_key = reasoner._normalize_cluster_key(f"repurposing_{repurposing_key}_concept")
        qa_description = reasoner.qa_space_descriptions.get(qa_key)
        feature_clause = None
        if qa_description:
            feature_clause = reasoner._description_to_feature_clause(qa_description)
            if feature_clause and repurposing_key == "shield":
                feature_clause = feature_clause.replace("width or length greater than height", "")
                feature_clause = (
                    feature_clause.replace("  ", " ")
                    .replace(" ,", ", ")
                    .strip(" ,")
                )
        if feature_clause:
            feature_clause = feature_clause.replace(",,", ",")
            feature_clause = feature_clause.replace(", ,", ", ")
            feature_clause = feature_clause.strip(" ,")
        steps: List[Dict[str, Any]] = [
            reasoner._create_reasoning_step(
                "question_analysis",
                analysis_sentence,
                input_data={"objects": object_set, "answer": answer},
            )
        ]
        if feature_clause:
            steps.append(
                reasoner._create_reasoning_step(
                    "question_analysis",
                    f"This concept expects {feature_clause}.",
                )
            )
        steps.append(
            reasoner._create_reasoning_step(
                "thinking",
                f"The {answer} offers the structure needed to accomplish the repurposed goal.",
                input_data={"object": answer},
                output_data={"answer": answer},
            )
        )
    else:
        steps = reasoner._generate_taxonomy_enhanced_reasoning(
            question_type, target_object, object_set, answer, "physical"
        )
    return steps
