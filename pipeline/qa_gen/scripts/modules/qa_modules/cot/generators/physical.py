"""
Physical property reasoning generator.
"""

from __future__ import annotations

from typing import Any, Dict, List

from modules.qa_modules.filter_utils import is_void_cluster


def generate_physical_property_reasoning(
    reasoner: "CoTReasoningGenerator",
    question_type: str,
    target_object: str,
    object_set: List[str],
    answer: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Generate structured reasoning for physical property questions."""
    del (question_type, target_object)
    property_name = kwargs.get("physical_property")
    cleaned_clusters: List[str] = []
    others = [obj for obj in object_set if obj != answer]
    if not others:
        others_clause = "the other objects"
        others_verb = "do"
    elif len(others) == 1:
        others_clause = others[0]
        others_verb = "does"
    else:
        others_clause = ", ".join(others)
        others_verb = "do"
    property_key = None
    visual_clues: List[str] = []
    positive_traits: List[str] = []
    negative_traits: List[str] = []

    if reasoner.taxonomy_utils:
        try:
            clusters = reasoner.taxonomy_utils.get_object_clusters(
                answer, "final_taxonomy_physical_properties"
            ) or []
            cleaned_clusters = [
                reasoner.clean_cluster_name(cluster)
                for cluster in clusters
                if cluster and not is_void_cluster(cluster, "physical")
            ]
        except Exception:  # noqa: BLE001
            cleaned_clusters = []

    if not property_name and cleaned_clusters:
        property_name = cleaned_clusters[0]

    if not property_name:
        property_name = "the specified physical property"

    property_key = reasoner._normalize_cluster_key(property_name)
    property_template = reasoner._property_feature_index.get(property_key) if property_key else None
    if isinstance(property_template, dict):
        visual_clues = property_template.get("visual_clues") or []
        positive_traits = (
            property_template.get("positive_traits")
            or property_template.get("traits")
            or []
        )
        negative_traits = property_template.get("negative_traits") or []

    steps: List[Dict[str, Any]] = []
    qa_filter_desc = kwargs.get("qa_filter_descriptor")
    if qa_filter_desc:
        steps.append(
            reasoner._create_reasoning_step(
                "analysis",
                f"I focus on objects already known for {qa_filter_desc}.",
            )
        )

    steps.append(
        reasoner._create_reasoning_step(
            "physical_properties",
            reasoner.clean_text(
                reasoner._compose_physical_property_analysis_sentence(
                    property_name,
                    visual_clues,
                    positive_traits,
                )
            ),
        )
    )

    if cleaned_clusters:
        group_phrases = [
            reasoner._render_domain_reference("physical_property", name)
            for name in cleaned_clusters[:2]
            if name
        ]
        group_summary = reasoner._summarize_feature_list(group_phrases)
        if group_summary:
            steps.append(
                reasoner._create_reasoning_step(
                    "physical_properties",
                    f"The {answer} consistently shows {group_summary}.",
                )
            )
    elif visual_clues or positive_traits:
        descriptive_bits: List[str] = []
        if visual_clues:
            descriptive_bits.append(
                f"visual cues such as {', '.join(visual_clues[:2])}"
            )
        if positive_traits:
            descriptive_bits.append(
                f"traits like {', '.join(positive_traits[:2])}"
            )
        if descriptive_bits:
            steps.append(
                reasoner._create_reasoning_step(
                    "physical_properties",
                    f"The {answer} shows {reasoner.clean_text(', '.join(descriptive_bits))}.",
                )
            )

    elimination_details: List[str] = []
    if reasoner.taxonomy_utils:
        for obj in object_set:
            if obj == answer:
                continue
            try:
                other_clusters = reasoner.taxonomy_utils.get_object_clusters(
                    obj, "final_taxonomy_physical_properties"
                ) or []
                filtered_clusters = [
                    cluster
                    for cluster in other_clusters
                    if cluster and not is_void_cluster(cluster, "physical")
                ]
                natural_names: List[str] = []
                feature_bits: List[str] = []
                for cluster in filtered_clusters:
                    natural_name = reasoner._naturalize_cluster_phrase(
                        cluster, "physical"
                    ) or reasoner.clean_cluster_name(cluster)
                    if natural_name:
                        natural_names.append(natural_name)
                    described = reasoner._describe_cluster_features(
                        "physical_property", cluster
                    )
                    if described:
                        feature_bits.append(described)
                detail = reasoner._summarize_feature_list(feature_bits)
                if detail:
                    elimination_details.append(
                        f"{obj} shows {detail}, which does not express {property_name.lower()}."
                    )
                elif natural_names:
                    elimination_details.append(
                        f"{obj} aligns with {', '.join(natural_names[:2])}, which does not express {property_name.lower()}."
                    )
                else:
                    elimination_details.append(
                        f"{obj} offers no physical traits that align with {property_name.lower()}."
                    )
            except Exception:  # noqa: BLE001
                elimination_details.append(
                    f"{obj} does not present evidence of {property_name}."
                )
    else:
        if negative_traits:
            elimination_details.append(
                f"{others_clause} {others_verb} show {', '.join(negative_traits[:2])}, which conflicts with {property_name.lower()}."
            )
        elimination_details.append(
            f"Objects other than {answer} do not demonstrate the physical traits tied to {property_name}."
        )

    if not elimination_details and negative_traits:
        elimination_details.append(
            f"{others_clause} {others_verb} show {', '.join(negative_traits[:2])}, which conflicts with {property_name.lower()}."
        )

    if elimination_details:
        steps.append(
            reasoner._create_reasoning_step(
                "elimination",
                "Other choices are ruled out because " + " ".join(elimination_details),
            )
        )

    thinking_clause = reasoner._compose_physical_property_thinking_sentence(
        answer,
        property_name,
        others_clause,
        others_verb,
        visual_clues,
        positive_traits,
    )
    steps.append(
        reasoner._create_reasoning_step(
            "thinking",
            thinking_clause,
        )
    )
    return steps
