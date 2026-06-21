"""
Material reasoning generator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _compose_material_thinking_sentence(
    reasoner: "CoTReasoningGenerator",
    answer: str,
    material_name: str,
    others_clause: str,
    others_verb: str,
    visual_summary: Optional[str],
    physical_summary: Optional[str],
    positive_summary: Optional[str],
    negative_summary: Optional[str],
    elimination_summary: Optional[str],
) -> str:
    base = f"The {answer} matches the {material_name} profile"
    descriptive_clauses: List[str] = []
    if visual_summary and physical_summary:
        descriptive_clauses.append(f"by showing {visual_summary} and feeling {physical_summary}")
    elif visual_summary:
        descriptive_clauses.append(f"by showing {visual_summary}")
    elif physical_summary:
        descriptive_clauses.append(f"with tactile traits like {physical_summary}")

    if positive_summary and not descriptive_clauses:
        descriptive_clauses.append(f"because it shares traits with {positive_summary}")
    elif positive_summary:
        descriptive_clauses.append(f"and aligns with materials such as {positive_summary}")

    sentence = base
    if descriptive_clauses:
        sentence += " " + ", ".join(descriptive_clauses).strip()

    if elimination_summary:
        contrast_clause = f", while {others_clause} {'do' if others_verb == 'are' else 'does'} show {elimination_summary}"
    elif negative_summary:
        contrast_clause = f", while {others_clause} {'do' if others_verb == 'are' else 'does'} relate to {negative_summary}"
    else:
        contrast_clause = f", whereas {others_clause} {'do' if others_verb == 'are' else 'does'} not."
    sentence += contrast_clause
    if not sentence.endswith("."):
        sentence += "."
    return reasoner.clean_text(sentence)


def _get_property_template_for_material_question(
    reasoner: "CoTReasoningGenerator",
    material_key: str,
) -> Optional[Dict[str, Any]]:
    """Get property template for material questions that don't match material templates."""
    property_mapping = {
        "sound_absorption": "sound_absorption",
        "flammability": "flammability",
        "thermal_touch": "thermal_touch",
        "scratch_resistance": "scratch_resistance",
        "latent_compressible": "latent_compressible",
        "electrical_conductivity": "electrical_conductivity",
    }

    for key, prop_type in property_mapping.items():
        if key in material_key:
            return reasoner.property_templates.get(prop_type)

    return None


def generate_material_reasoning(
    reasoner: "CoTReasoningGenerator",
    question_type: str,
    target_object: str,
    object_set: List[str],
    answer: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Generate structured material reasoning steps."""
    del target_object  # Unused in this context.
    material_key = question_type.replace("material_", "")
    others = [obj for obj in object_set if obj != answer]
    if not others:
        others_text = "the other objects"
        others_subject = "The other objects"
        others_verb = "are"
    elif len(others) == 1:
        others_text = others[0]
        others_subject = f"The {others[0]}"
        others_verb = "is"
    else:
        others_text = ", ".join(others)
        others_subject = f"The other options ({others_text})"
        others_verb = "are"
    others_clause = others_subject[:1].lower() + others_subject[1:] if others_subject else others_subject
    steps: List[Dict[str, Any]] = []
    visual_summary: Optional[str] = None
    physical_summary: Optional[str] = None
    positive_summary: Optional[str] = None
    negative_summary: Optional[str] = None
    elimination_feature_details: List[str] = []
    elimination_summary: Optional[str] = None

    qa_filter_desc = kwargs.get("qa_filter_descriptor")
    if qa_filter_desc:
        steps.append(
            reasoner._create_reasoning_step(
                "analysis",
                f"I focus on objects already noted for {qa_filter_desc}.",
            )
        )

    template_data = None
    for key, data in reasoner.material_templates.items():
        if key in material_key or any(word in material_key for word in key.split("_") if len(word) > 3):
            template_data = data
            break
    if not template_data:
        if any(token in material_key for token in ["textiles", "fibers", "leather"]):
            template_data = reasoner.material_templates.get("textiles_fibers_and_leather")
        elif "wood" in material_key or "plant" in material_key:
            template_data = reasoner.material_templates.get("wood_and_plant_based_solids")
        elif "metal" in material_key or "alloy" in material_key:
            template_data = reasoner.material_templates.get("metals_and_alloys")

    material_name = kwargs.get("material", material_key.replace("_", " ").strip())
    answer_entries = reasoner._collect_material_entries(answer)

    if template_data:
        visual_props = ", ".join(template_data["visual"][:2])
        physical_props = ", ".join(template_data["physical"][:2])
        visual_summary = visual_props or None
        physical_summary = physical_props or None
        steps.append(
            reasoner._create_reasoning_step(
                "material_properties",
                f"The {answer} shows material cues such as {visual_props} and feels {physical_props}.",
                output_data=answer,
                extra={"visual_properties": visual_props, "physical_properties": physical_props},
            )
        )
        positive_summary = visual_summary or physical_summary or positive_summary
    elif answer_entries:
        entry = answer_entries[0]
        visual_summary = ", ".join(entry.get("visual", [])[:2]) or None
        physical_summary = ", ".join(entry.get("physical", [])[:2]) or None
        descriptor_bits: List[str] = []
        if visual_summary:
            descriptor_bits.append(f"material cues such as {visual_summary}")
        if physical_summary:
            descriptor_bits.append(f"feels {physical_summary}")
        if descriptor_bits:
            steps.append(
                reasoner._create_reasoning_step(
                    "material_properties",
                    f"The {answer} shows {reasoner.clean_text(' and '.join(descriptor_bits))}.",
                    output_data=answer,
                )
            )
    else:
        steps.append(
            reasoner._create_reasoning_step(
                "material_properties",
                f"I examine material properties of each object to see which matches {material_name}.",
            )
        )

    if visual_summary and not positive_summary:
        positive_summary = visual_summary
    if physical_summary and not positive_summary:
        positive_summary = physical_summary

    if any(
        prop in material_key
        for prop in [
            "sound_absorption",
            "flammability",
            "thermal_touch",
            "scratch_resistance",
            "latent_compressible",
            "electrical_conductivity",
        ]
    ):
        prop_template = _get_property_template_for_material_question(reasoner, material_key)
        if prop_template:
            absorb_props = {k: ", ".join(v[:2]) for k, v in prop_template.items() if isinstance(v, list)}
            if "thermal_touch" in material_key:
                positive_summary = absorb_props.get("cool_materials")
                negative_summary = absorb_props.get("warm_materials")
            elif "sound_absorption" in material_key:
                positive_summary = absorb_props.get("absorbent_materials")
                negative_summary = absorb_props.get("reflective_materials")
            elif "scratch_resistance" in material_key:
                positive_summary = absorb_props.get("resistant_materials")
                negative_summary = absorb_props.get("soft_materials")
            elif "latent_compressible" in material_key:
                positive_summary = absorb_props.get("compressible_materials")
                negative_summary = absorb_props.get("rigid_materials")
            elif "flammability" in material_key:
                positive_summary = absorb_props.get("flammable_materials")
                negative_summary = absorb_props.get("non_flammable_materials")
            elif "electrical_conductivity" in material_key:
                positive_summary = absorb_props.get("conductive_materials")
                negative_summary = absorb_props.get("insulating_materials")
            positive_summary = (
                positive_summary
                or absorb_props.get("absorbent_materials")
                or absorb_props.get("resistant_materials")
                or absorb_props.get("warm_materials")
            )
            negative_summary = (
                negative_summary
                or absorb_props.get("reflective_materials")
                or absorb_props.get("soft_materials")
                or absorb_props.get("cool_materials")
            )

    elimination_sentences: List[str] = []
    for obj in others:
        object_entries = reasoner._collect_material_entries(obj)
        if object_entries:
            entry = object_entries[0]
            bits: List[str] = []
            if entry.get("visual"):
                bits.append(f"visual cues such as {', '.join(entry['visual'][:2])}")
            if entry.get("physical"):
                bits.append(f"traits like {', '.join(entry['physical'][:2])}")
            detail = reasoner._summarize_feature_list(bits) if bits else None
            label = entry.get("label") or obj
            if detail:
                elimination_feature_details.append(detail)
                elimination_sentences.append(f"{obj} shows {detail}, indicating {label} rather than {material_name}.")
            else:
                elimination_sentences.append(f"{obj} aligns with {label}, so it differs from {material_name}.")
        elif negative_summary:
            elimination_feature_details.append(negative_summary)
            elimination_sentences.append(f"{obj} relates to {negative_summary}, conflicting with {material_name}.")
        else:
            elimination_sentences.append(f"{obj} lacks distinctive material cues that match {material_name}.")

    if elimination_sentences:
        steps.append(
            reasoner._create_reasoning_step(
                "elimination",
                "Other choices are ruled out because " + " ".join(elimination_sentences),
                input_data={"objects": others},
            )
        )

    if elimination_feature_details:
        elimination_summary = reasoner._summarize_feature_list(elimination_feature_details)
        if elimination_summary and not negative_summary:
            negative_summary = elimination_summary

    steps.append(
        reasoner._create_reasoning_step(
            "thinking",
            _compose_material_thinking_sentence(
                reasoner,
                answer,
                material_name,
                others_clause,
                others_verb,
                visual_summary,
                physical_summary,
                positive_summary,
                negative_summary,
                elimination_summary,
            ),
            output_data=answer,
        )
    )
    return steps

