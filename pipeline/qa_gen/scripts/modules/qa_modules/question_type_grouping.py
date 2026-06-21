#!/usr/bin/env python3
"""
Question Type Grouping Utility

This module provides functions to map detailed question types to simplified
grouped labels for JSON output, while keeping the detailed types in code.

The mapping groups similar question types together:
- All affordance_* types → "affordance"
- All material_* types → "material"
- All spatial_* types → "spatial"
etc.
"""

from typing import Dict, List, Optional


class QuestionTypeGrouper:
    """Maps detailed question types to simplified grouped labels"""
    
    def __init__(self):
        # Define grouping mappings: detailed_type -> simplified_label
        self.grouping_map = {
            # Taxonomy Description questions
            "material_property": "taxonomy_description",
            "physical_property": "taxonomy_description",
            "function_knowledge": "taxonomy_description",
            "description_matching": "taxonomy_description",
            "compositional_set_subtraction_hollow": "taxonomy_description",
            "physical properties": "taxonomy_description",
            "physical_properties": "taxonomy_description",
            **{f"affordance_{suffix}": "taxonomy_description" for suffix in [
                "furniture",
                "contain__carry__package",
                "grip__carry__operate",
                "operate__use_device",
                "mechanical_control",
                "mediated_action_and_meaning",
                "food_—_ingredients_and_produce",
                "food_—_prepared_dishes",
                "cleaning_and_sanitation",
                "control__express__light",
                "grow__plant_(vegetation)",
                "enclosures_and_venues_(enter_use)",
                "place__support__work_on",
                "architectural_components_and_fixtures",
                "art_display_(view_appraise)",
                "build__span__occupy",
                "display__exhibit__signal_value",
                "household__facility_operations",
                "interact_with_living_moving_things",
                "sit__ride__attend",
                "structured_operational_engagement",
                "tableware_and_serveware",
                "wearables_and_apparel",
            ]},

            # Taxonomy Reasoning questions
            "flammability": "taxonomy_reasoning",
            "functional_seating": "taxonomy_reasoning",
            "functional_foldable": "taxonomy_reasoning",
            "repurposing_shield_concept": "taxonomy_reasoning",
            "repurposing_container_concept": "taxonomy_reasoning",
            "repurposing_reflector_concept": "taxonomy_reasoning",
            "repurposing_cushion_concept": "taxonomy_reasoning",
            "repurposing_stepstool_concept": "taxonomy_reasoning",
            "repurposing_bookend_concept": "taxonomy_reasoning",
            "material_sound_absorption": "taxonomy_reasoning",
            "material_thermal_touch": "taxonomy_reasoning",
            "material_scratch_resistance": "taxonomy_reasoning",
            "counterfactual_water": "taxonomy_reasoning",
            "counterfactual_heat": "taxonomy_reasoning",
            "latent_containment": "taxonomy_reasoning",
            "latent_compressible": "taxonomy_reasoning",
            "compositional_set_subtraction_container": "taxonomy_reasoning",

            # Spatial Relation questions
            "spatial_left_right": "spatial_relation",
            "spatial_above_below": "spatial_relation",
            "spatial_front_behind": "spatial_relation",
            "spatial_closer_to_camera": "spatial_relation",
        }
        
        # Alternative: More granular grouping (optional)
        self.granular_grouping_map = {
            # Keep affordance subtypes but group similar ones
            "affordance_furniture": "affordance_furniture",
            "affordance_sit__ride__attend": "affordance_furniture",
            "affordance_contain__carry__package": "affordance_container",
            "affordance_grip__carry__operate": "affordance_portable",
            "affordance_operate__use_device": "affordance_operable",
            # ... etc
        }
    
    def get_simplified_type(self, detailed_type: str, use_granular: bool = False) -> str:
        """
        Get simplified question type label for JSON output.
        
        Args:
            detailed_type: The detailed question type (e.g., "affordance_furniture")
            use_granular: If True, use more granular grouping. If False, use simple grouping.
        
        Returns:
            Simplified label (e.g., "affordance" or "affordance_furniture" if granular)
        """
        if use_granular:
            return self.granular_grouping_map.get(detailed_type, detailed_type)
        else:
            # First check explicit mapping
            if detailed_type in self.grouping_map:
                return self.grouping_map[detailed_type]
            
            # Fallback: Use prefix-based grouping for unknown types
            if detailed_type.startswith('affordance_'):
                return "taxonomy_description"
            elif detailed_type.startswith('material_'):
                return "taxonomy_description"
            elif detailed_type.startswith('spatial_'):
                return "spatial_relation"
            elif detailed_type.startswith('repurposing_'):
                return "taxonomy_reasoning"
            elif detailed_type.startswith('counterfactual_'):
                return "taxonomy_reasoning"
            elif detailed_type.startswith('latent_'):
                return "taxonomy_reasoning"
            elif detailed_type.startswith('functional_'):
                return "taxonomy_reasoning"
            elif detailed_type.startswith('compositional_'):
                # Default compositional variants to taxonomy_reasoning unless explicitly mapped
                return "taxonomy_reasoning"
            
            # If no prefix matches, return original type
            return detailed_type
    
    def get_all_groupings(self) -> Dict[str, List[str]]:
        """
        Get reverse mapping: simplified_label -> list of detailed types.
        
        Returns:
            Dictionary mapping simplified labels to lists of detailed question types
        """
        reverse_map = {}
        for detailed_type, simplified_label in self.grouping_map.items():
            if simplified_label not in reverse_map:
                reverse_map[simplified_label] = []
            reverse_map[simplified_label].append(detailed_type)
        return reverse_map
    
    def get_category_statistics(self, question_types: List[str]) -> Dict[str, int]:
        """
        Count questions by simplified category.
        
        Args:
            question_types: List of detailed question types
        
        Returns:
            Dictionary with simplified category counts
        """
        stats = {}
        for qtype in question_types:
            simplified = self.get_simplified_type(qtype)
            stats[simplified] = stats.get(simplified, 0) + 1
        return stats


# Global instance
question_type_grouper = QuestionTypeGrouper()


def get_simplified_question_type(detailed_type: str, use_granular: bool = False) -> str:
    """
    Convenience function to get simplified question type.
    
    Args:
        detailed_type: The detailed question type
        use_granular: Use granular grouping (False = simple grouping)
    
    Returns:
        Simplified question type label
    """
    return question_type_grouper.get_simplified_type(detailed_type, use_granular)


