#!/usr/bin/env python3
"""
Centralized Question Templates

This module provides standardized question templates for all QA generation scripts
to ensure consistency across synthetic scenes, real images, and question space analysis.
"""

import random

from typing import Dict, Any, Optional

FORCE_LAST_VARIANT_TYPES = {
    'affordance_furniture',
    'affordance_contain__carry__package',
    'affordance_grip__carry__operate',
    'affordance_operate__use_device',
    'affordance_mechanical_control',
    'affordance_mediated_action_and_meaning',
    'affordance_food_—_ingredients_and_produce',
    'affordance_food_—_prepared_dishes',
    'affordance_cleaning_and_sanitation',
    'affordance_control__express__light',
    'affordance_grow__plant_(vegetation)',
    'affordance_enclosures_and_venues_(enter_use)',
    'affordance_place__support__work_on',
}


class QuestionTemplates:
    """Centralized question templates for all QA types"""
    
    def __init__(self):
        self.templates = {
            # Functional questions
            'functional_seating': "Which object can be used for sitting or resting?",
            'functional_foldable': "Which object can be folded or collapsed to save space?",
            'flammability': "Which object poses the highest fire risk?",
            
            # Material questions (material_property is the only active material question type)
            # Legacy material variant questions (kept for backward compatibility, but filtered out in generation):
            'material_sound_absorption': [
                "Which object best absorbs sound?", 
                "There are some loud noises aroung and you want to cover you ears, which object would be the best to do so?"
            ],
            'material_thermal_touch': ["Which object would feel coldest to touch in a cold room?", ],
            'material_scratch_resistance': ["Which surface is least likely to have scratch marks if you scratch it with a fingernail?"],
            # 'material_metals_and_alloys': "Which object is made of metal?",
            # 'material_textiles_fibers_and_leather': "Which object is made of fabric or leather?",
            # 'material_wood_and_plant_based_solids': "Which object is made of wood?",
            # 'material_plastics_rubber_and_polymers': "Which object is made of plastic?",
            # 'material_paper_cardboard_and_pulp': "Which object is made of paper or cardboard?",
            # 'material_glass_and_transparent': "Which object is made of glass?",
            # 'material_glass_and_transparent_(silicate)': "Which object is made of glass?",
            # 'material_biological_(plants/flowers)': "Which object is a plant or flower?",
            # 'material_organic_food_and_edible_matter': "Which object is edible food?",
            # Active material question (uses actual material string from object annotations)
            'material_property': [
                "Which object is made of '{material}'?",
                "Which object seems to be made of {material}, judging by its texture or surface appearance?"
            ],
            'physical_property': [
                "Which object shows the {property} physical property in this scene?",
                "Which object appears to exhibit the {property} physical characteristic here?",
                "Which object seems to have the {property} physical trait?"
            ],
            
            # Affordance questions
            'affordance_furniture': [
                "Which object is furniture?",
                "Which of these objects is likely to be considered furniture?"
            ],
            'affordance_contain__carry__package': [
                "Which object can contain or carry items?",
                "Which object looks like it could hold or carry other items inside it?"
            ],
            'affordance_grip__carry__operate': [
                "Which object can be gripped and carried?",
                "Which object seems small or shaped for someone to grip and carry easily?"
            ],
            'affordance_operate__use_device': [
                "Which object is a device or tool that can be operated without power source?",
                "Which object could be operated by hand without needing electricity or power?"
            ],
            'affordance_mechanical_control': [
                "Which object requires mechanical control and power source to operate?",
                "Which object looks like a powered machine that requires mechanical control to operate?"
            ],
            'affordance_mediated_action_and_meaning': [
                "Which object involves reading or communication?",
                "Which object seems meant for reading, writing, or communication?"
            ],
            'affordance_food_—_ingredients_and_produce': [
                "Which object is food or produce?",
                "Which object appears to be natural food or fresh produce?"
            ],
            'affordance_food_—_prepared_dishes': [
                "Which object is prepared food?",
                "Which object looks like a prepared meal ready to eat?"
            ],
            'affordance_cleaning_and_sanitation': [
                "Which object is used for cleaning?",
                "Which object seems intended for cleaning or wiping surfaces?"
            ],
            'affordance_control__express__light': [
                "Which object controls or produces light?",
                "Which object could produce or control light in this scene?"
            ],
            'affordance_grow__plant_(vegetation)': [
                "Which object is used for growing plants?",
                "Which object looks like it is used to hold or support growing plants?"
            ],
            'affordance_enclosures_and_venues_(enter_use)': [
                "Which object is an enclosed space or venue?",
                "Which object appears to be an enclosed space or shelter someone could enter?"
            ],
            'affordance_place__support__work_on': [
                "Which object can have items placed on it?",
                "Which object seems flat and sturdy enough to place items on?"
            ],
            
            # Repurposing questions (only templates that are in QA space)
            'repurposing_shield_concept': [
                "What object in the scene can someone grab and repurpose as a shield to block danger?",
                "If a person suddenly needs protection from debris or danger, which object here could be used as a shield?",
                "Imagine a sudden impact toward you—what item in this scene could you hold to shield yourself?",
                "During an unexpected emergency, which object could be quickly repurposed as a makeshift shield?",
                "If someone must defend themselves or block projectiles in a pinch, which item could serve as a shield?"
            ],
            'repurposing_container_concept': [
                "If you had to collect or carry something quickly, which object here could act as a makeshift container?",
                "What item in this scene could be repurposed to store water, sand, or small supplies in an emergency?",
                "Suppose someone needs an improvised vessel—what object could be used as a container?",
                "If no bag or box is available, which object could temporarily hold or transport items?",
                "When you must improvise a storage solution, which item in the scene could serve as a container?"
            ],
            'repurposing_reflector_concept': [
                "If you needed to reflect light or signal for help, which object here could be repurposed as a reflector?",
                "Imagine the sun or a flashlight shining—what item in the scene would best bounce that light?",
                "In an emergency signalling situation, which object could act as a reflective surface?",
                "Suppose you wanted to redirect sunlight toward a darker spot—what could serve as a temporary reflector?",
                "If you must illuminate a shadowed area, which nearby object could be used as a reflector?"
            ],
            'repurposing_cushion_concept': [
                "If comfort is needed, which object could be rearranged or stuffed to work as a cushion?",
                "Imagine needing something soft to sit on—what item here could be repurposed as a cushion?",
                "If someone wants a temporary pillow, which object could be folded or rolled up for cushioning?",
                "Suppose you need padding while kneeling or resting—what object could serve as a makeshift cushion?",
                "When a fragile item must be protected, which scene object could be used as cushioning material?"
            ],
            'repurposing_stepstool_concept': [
                "If you had to reach a higher shelf, which object could double as a stepstool?",
                "Imagine changing a light bulb with no ladder—what item here could provide enough height?",
                "When you need a quick boost, which object in this scene could work as a stepstool?",
                "Suppose there’s no stool available—which sturdy object could someone stand on to reach upward?",
                "If you must improvise to access something high, which object could function as a temporary stepstool?"
            ],
            'repurposing_bookend_concept': [
                "If you needed to keep books upright, which object could act as a bookend?",
                "Imagine organizing a shelf without real bookends—what nearby item could hold books in place?",
                "If proper bookends are missing, which object here could support books and prevent them from falling?",
                "Suppose you want to stabilize a row of books—what could be repurposed as a bookend?",
                "When books start tipping over, which heavy or supportive object could keep them upright?"
            ],
            
            # Counterfactual questions
            'counterfactual_water': [
                "If someone accidentally spills water here, which object would be damaged first?",
                "Imagine a glass tips over—what item in this scene is most vulnerable to water damage?",
                "Suppose the floor floods—what object would suffer immediately from water exposure?",
                "If liquid splashes across the area, which item would be ruined or malfunction first?",
                "During a leak or spill, which object would get soaked and stop working before the others?"
            ],
            'counterfactual_heat': [
                "If the temperature suddenly spikes, which object here would be most affected by heat?",
                "Imagine intense sunlight or fire exposure—what item would melt, warp, or burn first?",
                "Suppose this scene is placed near a strong heat source—what object would deteriorate most quickly?",
                "In extreme heat, which item would react visibly before the others?",
                "If a sudden heatwave hits, which object in the scene would be damaged first?"
            ],
            
            # Latent state questions
            'latent_containment': "Which object can hide small items while keeping the area tidy?",
            'latent_compressible': "Which object can be compressed to fit in tight spaces without damaging its structure?",
            
            # Compositional questions
            'compositional_set_subtraction_container': [
                "Which object is solid and movable but clearly not intended to hold other things?",
                "Identify the item that is rigid and portable, yet not meant to function as a container.",
                "In this scene, which object can be carried around but isn’t designed to store items?",
                "Which object here is firm and transportable but not supposed to act as a vessel?",
                "Among the items shown, which one is sturdy and movable but definitely not a container?"
            ],
            'compositional_set_subtraction_hollow': [
                "Which object is hollow?",
                "Which object looks hollow inside based on its visible shape or openings?"
            ],
            
            # Advanced compositional questions
            'compositional_counterfactual_material': "If {target} were made of {to_material} instead of {from_material}, how would it change?",
            'compositional_action_based': "Use {target} to {goal}. How?",
            'compositional_set_subtraction_dynamic': "Which object is {description}? Options: {options}",
            
            
            'description_matching': "Which object matches this description: '{description}'?",
            'function_knowledge': [
                "Which object is used as '{function}'?",
                "Which object looks like it is designed for {function} in this scene?"
            ],
            
            # Spatial questions
            
            'spatial_left_right': "Is {object1} to the left or right of {object2}? Options: left, right.",
            'spatial_above_below': "Is {object1} above or below {object2}? Options: above, below.",
            'spatial_front_behind': "Is {object1} in front of or behind {object2}? Options: front, behind.",
            'spatial_closer_to_camera': "Is {object1} or {object2} closer to the camera?",
            
        }
    
    def get_template(self, question_type: str, **kwargs) -> str:
        """Get question template for a specific question type"""
        if question_type in self.templates:
            template = self.templates[question_type]
            if isinstance(template, list):
                if question_type in FORCE_LAST_VARIANT_TYPES:
                    selected = template[-1]
                else:
                    selected = random.choice(template)
                if kwargs:
                    try:
                        return selected.format(**kwargs)
                    except KeyError:
                        return selected
                return selected
            if kwargs:
                try:
                    return template.format(**kwargs)
                except KeyError:
                    return template
            return template
        
        # Fallback for unknown question types
        return self._generate_fallback_template(question_type)
    
    def _generate_fallback_template(self, question_type: str) -> str:
        """Generate fallback template for unknown question types"""
        if "material" in question_type.lower():
            material_type = question_type.replace('material_', '').replace('_', ' ')
            return f"Which object is made of {material_type}?"
        elif "affordance" in question_type.lower():
            affordance_type = question_type.replace('affordance_', '').replace('_(', ' (').replace('__', ' ')
            # Handle special cases
            if 'enclosures_and_venues' in affordance_type or 'enter_use' in affordance_type:
                return "Which object is an enclosed space or venue?"
            elif 'enter_use' in affordance_type:
                return "Which object can be entered?"
            return f"Which object has the affordance of {affordance_type}?"
        elif "function" in question_type.lower():
            function_type = question_type.replace('function_', '').replace('_', ' ')
            return f"Which object has the function of {function_type}?"
        elif "compositional" in question_type.lower():
            return f"Which object has the specified compositional properties?"
        elif "repurposing" in question_type.lower():
            return f"Which object could be repurposed?"
        elif "counterfactual" in question_type.lower():
            return f"Which object would be affected in this scenario?"
        elif "latent" in question_type.lower():
            return f"Which object has hidden properties?"
        else:
            return f"Which object has the specified properties?"
    
    def get_all_templates(self) -> Dict[str, str]:
        """Get all available question templates"""
        return self.templates.copy()
    
    def add_template(self, question_type: str, template: str) -> None:
        """Add a new question template"""
        self.templates[question_type] = template
    
    def has_template(self, question_type: str) -> bool:
        """Check if template exists for question type"""
        return question_type in self.templates


# Global instance for easy access
question_templates = QuestionTemplates()


def get_question_template(question_type: str, **kwargs) -> str:
    """Convenience function to get question template"""
    return question_templates.get_template(question_type, **kwargs)


def get_all_question_templates() -> Dict[str, str]:
    """Convenience function to get all templates"""
    return question_templates.get_all_templates()
