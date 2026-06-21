#!/usr/bin/env python3
"""
Comprehensive Chain-of-Thought Reasoning Utilities

This module provides sophisticated reasoning generation for all QA types, combining
the best features from the deprecated reasoning scripts into a unified utility.
"""

import json
import math
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.qa_modules.compositional_utils import CompositionalUtils
from modules.qa_modules.filter_utils import is_void_cluster
from modules.qa_modules.cot import core

logger = logging.getLogger(__name__)

class CoTReasoningGenerator:
    STEP_TYPE_NORMALIZATION = {
        "repurposing_analysis": "question_analysis",
        "repurposing_traits": "concept_analysis",
        "material_analysis": "attribute_analysis",
        "material_properties": "attribute_analysis",
        "affordance_analysis": "concept_analysis",
        "composition_analysis": "question_analysis",
        "description_analysis": "question_analysis",
        "functional_analysis": "question_analysis",
        "taxonomy_filter": "filter_analysis",
        "qa_filter": "filter_analysis",
        "orientation": "spatial_analysis",
        "calculation": "spatial_calculation",
    }
    """Comprehensive Chain-of-Thought reasoning generator for all question types"""
    
    def __init__(self, taxonomy_utils=None):
        self.taxonomy_utils = taxonomy_utils
        self.compositional_utils = CompositionalUtils(taxonomy_utils)
        self._last_spatial_answer_label: Optional[str] = None
        # Comprehensive material property mappings for ALL material types
        self.material_templates = {
            'metals_and_alloys': {
                'visual': ['shiny metallic luster', 'reflective surface', 'silver or gray coloration', 'smooth polished finish'],
                'physical': ['hard and rigid', 'heavy weight', 'conducts heat and electricity', 'durable and strong'],
                'context': ['vehicles', 'tools', 'structural elements', 'appliances'],
                'reasoning_template': "Metal objects have {visual_props} and feel {physical_props}. Among {objects}, the {answer} shows metallic characteristics like reflective surfaces and rigid structure. Other objects lack these metal properties and appear to be made of different materials."
            },
            'glass_and_transparent': {
                'visual': ['transparent or translucent appearance', 'smooth glossy surface', 'clear or colored visible through material', 'reflective when polished'],
                'physical': ['brittle and fragile', 'smooth to touch', 'non-conductive', 'lightweight'],
                'context': ['windows', 'containers', 'decorative objects', 'optical devices'],
                'reasoning_template': "Glass objects are {visual_props} and feel {physical_props}. Among {objects}, the {answer} displays clear glass properties: transparency allowing light through, smooth reflective surface, and brittle material characteristics. Other objects are opaque solid materials."
            },
            'textiles_fibers_and_leather': {
                'visual': ['woven or knitted texture visible on surface', 'flexible and draping appearance', 'fabric patterns and fibers', 'soft surface'],
                'physical': ['flexible and bendable when handled', 'soft to touch', 'absorbent material', 'lightweight'],
                'context': ['clothing', 'furniture upholstery', 'bags', 'decorative items'],
                'reasoning_template': "Textile objects show {visual_props} and feel {physical_props}. Among {objects}, the {answer} exhibits fabric characteristics: flexible material that can bend and fold, visible fabric texture or pattern, and soft draping qualities. Other objects are rigid solid materials."
            },
            'wood_and_plant_based_solids': {
                'visual': ['natural grain patterns and texture', 'warm brown tones or natural colors', 'organic wood-like surface appearance', 'matte or slightly glossy finish'],
                'physical': ['moderately hard surface', 'natural warmth to touch', 'can be carved', 'biodegradable'],
                'context': ['furniture', 'construction', 'decorative items', 'tools'],
                'reasoning_template': "Wooden objects show {visual_props} and feel {physical_props}. Among {objects}, the {answer} has wooden characteristics: visible wood grain texture, natural brown or wood-colored appearance, and wood-like structural quality. Other objects show synthetic or different material properties."
            },
            'ceramics_porcelain_and_earthenware': {
                'visual': ['smooth glazed surface', 'matte or glossy ceramic finish', 'often white or colored ceramic appearance', 'hard non-porous surface'],
                'physical': ['hard and brittle material', 'non-porous when glazed', 'heat resistant', 'fragile'],
                'context': ['dishes', 'decorative objects', 'tiles', 'containers'],
                'reasoning_template': "Ceramic objects have {visual_props} and feel {physical_props}. Among {objects}, the {answer} shows ceramic properties: smooth glazed surface typical of ceramics, hard brittle material, and ceramic-like finish. Other objects are made of different materials."
            },
            'plastics_rubber_and_polymers': {
                'visual': ['smooth synthetic surface', 'bright synthetic colors', 'uniform molded texture', 'manufactured appearance'],
                'physical': ['lightweight synthetic material', 'flexible or rigid depending on type', 'non-conductive', 'durable'],
                'context': ['containers', 'toys', 'electronics', 'household items'],
                'reasoning_template': "Plastic objects have {visual_props} and feel {physical_props}. Among {objects}, the {answer} shows plastic characteristics: synthetic molded appearance, bright colors typical of plastics, and synthetic material properties. Other objects are made of natural or different materials."
            },
            'paper_cardboard_and_pulp': {
                'visual': ['flat matte surface', 'layered sheet edges', 'printed graphics or text', 'fibrous texture'],
                'physical': ['lightweight and flexible', 'slightly rigid yet bendable', 'compressible when pressed', 'non-reflective surface'],
                'context': ['posters', 'boxes', 'packaging materials', 'paper products'],
                'reasoning_template': "Paper-based objects show {visual_props} and feel {physical_props}. Among {objects}, the {answer} exhibits paper and cardboard traits with visible sheet layers, matte finish, and lightweight bendable structure. Other objects are made from harder or more reflective materials."
            },
            'stone_concrete_and_mineral': {
                'visual': ['rough or smooth surface', 'gray or natural colors', 'solid appearance', 'textured finish'],
                'physical': ['very hard', 'heavy weight', 'durable', 'non-conductive'],
                'context': ['construction', 'decorative items', 'tools', 'architectural elements'],
                'reasoning_template': "Stone objects have {visual_props} and feel {physical_props}. Among {objects}, the {answer} shows stone characteristics while others are made of different materials."
            },
            'biological_animals_body_parts': {
                'visual': ['organic texture', 'natural colors', 'irregular shapes', 'biological patterns'],
                'physical': ['flexible or rigid', 'organic feel', 'biodegradable', 'natural materials'],
                'context': ['food', 'decorative items', 'tools', 'clothing'],
                'reasoning_template': "Biological objects show {visual_props} and have {physical_props}. Among {objects}, the {answer} exhibits biological characteristics while others are synthetic."
            },
            'composites_and_multi_material_products': {
                'visual': ['mixed textures', 'layered appearance', 'complex surface', 'multiple materials visible'],
                'physical': ['combination of properties', 'engineered feel', 'specific design', 'purpose-built'],
                'context': ['specialized equipment', 'high-tech items', 'composite structures', 'engineered products'],
                'reasoning_template': "Composite objects show {visual_props} and have {physical_props}. Among {objects}, the {answer} exhibits composite characteristics while others are single-material."
            }
        }
        
        # Comprehensive affordance templates
        self.affordance_templates = {
            'sit_ride_attend': {
                'features': ['a flat horizontal surface at appropriate height', 'stable base and structure', 'weight-bearing capacity'],
                'requirements': ['sufficient size for seating', 'sturdy construction to support weight', 'accessibility'],
                'reasoning_template': "Seating objects provide {features} for {requirements}. Among {objects}, the {answer} has these observable seating characteristics: a flat surface for sitting, structural support, and appropriate dimensions for a person. Other objects lack these specific attributes."
            },
            'contain_carry_package': {
                'features': ['has a hollow center to put stuff in it', 'a hollow interior with visible opening', 'enclosed space for containment', 'portable structure with handles or carrying capability'],
                'requirements': ['storage space to hold items', 'accessibility through openings', 'ability to be transported'],
                'reasoning_template': "Container objects have {features} for {requirements}. Among {objects}, the {answer} shows container characteristics: an enclosed hollow center to put stuff in, with an opening for placing and retrieving items. Other objects are solid structures without containment capability."
            },
            'wearables_and_apparel': {
                'features': ['flexible fabric or soft material', 'human body proportions', 'fastening mechanisms like buttons, zippers, or ties'],
                'requirements': ['fit around human body', 'comfortable to wear', 'suitable for clothing purposes'],
                'reasoning_template': "Wearable objects have {features} for {requirements}. Among {objects}, the {answer} has wearable features: flexible material suitable for covering the body, with dimensions and shape designed for wearing. Other objects are rigid or not designed for wearing."
            },
            'build_span_occupy': {
                'features': ['load-bearing structural elements', 'rigid construction materials', 'designed for long-term installation'],
                'requirements': ['structural support and stability', 'durability for permanent use', 'connection to other structural elements'],
                'reasoning_template': "Building objects have {features} for {requirements}. Among {objects}, the {answer} has building features: rigid structural construction designed to support loads and occupy space permanently. Other objects are temporary or lack structural building characteristics."
            },
            'interact_with_living_moving_things': {
                'features': ['interactive elements or controls', 'movement mechanisms or motors', 'responsive design for engagement'],
                'requirements': ['capability for interaction', 'movement or response capability', 'engagement with users'],
                'reasoning_template': "Interactive objects have {features} for {requirements}. Among {objects}, the {answer} shows interactive features: mechanisms for movement, control interfaces, or responsive elements designed for user interaction. Other objects are static and non-interactive."
            },
            'view_read_appraise': {
                'features': ['a flat display surface', 'readable text, images, or visual content', 'appropriate dimensions for viewing'],
                'requirements': ['information display capability', 'visibility of content', 'clear presentation'],
                'reasoning_template': "Display objects have {features} for {requirements}. Among {objects}, the {answer} has display features: a visible surface with text, images, or information designed for viewing. Other objects lack display surfaces or visual content."
            },
            'clean_sanitize': {
                'features': ['absorbent or cleaning surfaces', 'textured or soft materials for scrubbing', 'hygienic properties for sanitation'],
                'requirements': ['cleaning and sanitation capability', 'absorption or scrubbing capacity', 'hygiene maintenance'],
                'reasoning_template': "Cleaning objects have {features} for {requirements}. Among {objects}, the {answer} has cleaning features: materials and surfaces designed for cleaning, scrubbing, or sanitizing purposes. Other objects lack cleaning or sanitation capabilities."
            },
            'prepare_cook': {
                'features': ['heat-resistant materials for cooking', 'appropriate shape for food preparation', 'cooking surfaces or vessels'],
                'requirements': ['cooking and food preparation capability', 'heat resistance', 'food handling safety'],
                'reasoning_template': "Cooking objects have {features} for {requirements}. Among {objects}, the {answer} has cooking features: materials and design suitable for food preparation, cooking, or heating food. Other objects lack the heat-resistant surfaces and food preparation capabilities."
            },
            'furniture': {
                'features': ['furniture-related affordances like seating, storage, or support', 'sturdy construction with appropriate height and structure'],
                'requirements': ['furniture functionality such as sitting, storing, or supporting', 'suitable materials and stable design'],
                'reasoning_template': "Furniture objects provide {features} for {requirements}. Among {objects}, the {answer} is designed as furniture with surfaces, structures, and dimensions suited for sitting, sleeping, storing, or supporting household activities. Other objects lack furniture characteristics."
            },
            'architectural_components_and_fixtures': {
                'features': ['architectural integration into buildings', 'permanent installation characteristics', 'functional building components'],
                'requirements': ['integrated into architectural structures', 'permanent fixtures with specific functions', 'building infrastructure elements'],
                'reasoning_template': "Architectural fixtures have {features} for {requirements}. Among {objects}, the {answer} shows architectural fixture features: permanent installation into building structures with specific functional purposes. Other objects are temporary or not architectural fixtures."
            },
            'art_display_(view_appraise)': {
                'features': ['artwork or aesthetic display', 'framed or mounted presentation', 'artistic or decorative content'],
                'requirements': ['artistic value and display', 'presentation for viewing and appraisal', 'aesthetic content'],
                'reasoning_template': "Art display objects have {features} for {requirements}. Among {objects}, the {answer} is designed as artwork or decorative display with aesthetic content intended for viewing. Other objects lack artistic or decorative display characteristics."
            },
            'cleaning_and_sanitation': {
                'features': ['cleaning tools or supplies', 'hygienic materials', 'sanitation equipment'],
                'requirements': ['cleaning and sanitizing capability', 'hygiene maintenance', 'sanitation purposes'],
                'reasoning_template': "Cleaning objects have {features} for {requirements}. Among {objects}, the {answer} is specifically designed for cleaning or sanitation purposes with appropriate materials and structure. Other objects lack cleaning or sanitation capabilities."
            },
            'control__express__light': {
                'features': ['light production or control mechanisms', 'illumination capability', 'lighting controls'],
                'requirements': ['light generation or control', 'illumination and visibility', 'light management'],
                'reasoning_template': "Lighting objects have {features} for {requirements}. Among {objects}, the {answer} is designed to produce, control, or express light with visible light sources or lighting mechanisms. Other objects lack lighting capabilities."
            },
            'display__exhibit__signal_value': {
                'features': ['exhibition or display functionality', 'signal or communicate information', 'showcase or present value'],
                'requirements': ['display and exhibition capability', 'information or value signaling', 'presentation purposes'],
                'reasoning_template': "Display objects have {features} for {requirements}. Among {objects}, the {answer} is designed to exhibit, signal, or display information and value. Other objects lack exhibition or display capabilities."
            },
            'enclosures_and_venues_(enter_use)': {
                'features': ['enclosed spaces or venues', 'accessible entry and interior space', 'designed for entering and using'],
                'requirements': ['venue or enclosure functionality', 'accessibility for entry and use', 'enclosed usable space'],
                'reasoning_template': "Venue objects have {features} for {requirements}. Among {objects}, the {answer} provides enclosed spaces or venues designed for entering and using. Other objects lack the enclosed venue characteristics."
            },
            'food_—_ingredients_and_produce': {
                'features': ['edible natural ingredients', 'raw food or produce', 'fresh agricultural products'],
                'requirements': ['edibility and natural food characteristics', 'ingredient quality', 'produce characteristics'],
                'reasoning_template': "Food ingredient objects have {features} for {requirements}. Among {objects}, the {answer} is natural edible produce or ingredients suitable for consumption or cooking. Other objects are processed items or non-food materials."
            },
            'food_—_prepared_dishes': {
                'features': ['prepared and ready-to-eat food', 'cooked or processed dishes', 'serving-ready presentation'],
                'requirements': ['prepared food characteristics', 'cooked or processed dishes', 'serving-ready food'],
                'reasoning_template': "Prepared food objects have {features} for {requirements}. Among {objects}, the {answer} is prepared and cooked food ready to eat or serve. Other objects are raw ingredients or non-food materials."
            },
            'contain__carry__package': {
                'features': ['hollow interior for containing items', 'opening or closure mechanism', 'carrying or packaging capability'],
                'requirements': ['containment capability', 'packaging and carrying functionality', 'storage capacity'],
                'reasoning_template': "Containment objects have {features} for {requirements}. Among {objects}, the {answer} can contain, carry, or package items with enclosed space and appropriate openings. Other objects lack containment or packaging capabilities."
            },
            'grip__carry__operate': {
                'features': ['grippable handles or structures', 'manually operable size and shape', 'portable with grasping capability'],
                'requirements': ['ability to be gripped and held', 'manual operation capability', 'carrying portability'],
                'reasoning_template': "Grippable objects have {features} for {requirements}. Among {objects}, the {answer} has structures designed to be gripped, carried, and operated by hand with appropriate size, shape, and handles. Other objects are too large, immovable, or lack grip capability."
            },
            'operate__use_device': {
                'features': ['device or tool functionality', 'operable controls or interfaces', 'user-operated devices'],
                'requirements': ['capability for operation as a device or tool', 'interactive control elements', 'device or tool operation'],
                'reasoning_template': "Devices and tools have {features} for {requirements}. Among {objects}, the {answer} is a device or tool that can be operated: such as appliances, electronics, instruments, or tools designed for user operation. Other objects are not devices or tools (e.g., machinery, structural elements, or natural objects)."
            },
            'mechanical_control': {
                'features': ['requires mechanical control to operate', 'vehicles or machinery needing control', 'mechanical operation requirement'],
                'requirements': ['mechanical control dependency', 'vehicle or machinery classification', 'requires operator control'],
                'reasoning_template': "Objects requiring mechanical control have {features} for {requirements}. Among {objects}, the {answer} requires mechanical control to operate: such as vehicles (car, bus, motorbike), machinery, or systems that need an operator to control their motion, power, or operation. Other objects do not require mechanical control (e.g., simple tools, devices, or static objects)."
            },
            'mediated_action_and_meaning': {
                'features': ['readable text or symbols', 'communication or informational display', 'text-based or symbolic content'],
                'requirements': ['reading and communication capability', 'information conveyance', 'symbolic or textual meaning'],
                'reasoning_template': "Communication objects have {features} for {requirements}. Among {objects}, the {answer} has communication features: readable text, symbols, or information displays designed for reading and conveying meaning. Other objects lack textual or communication content."
            },
            'grow__plant_(vegetation)': {
                'features': ['plant containers or soil-holding structures', 'water drainage systems', 'plant support and growing space'],
                'requirements': ['capability for plant cultivation', 'suitable for vegetation growth', 'plant care functionality'],
                'reasoning_template': "Plant growing objects have {features} for {requirements}. Among {objects}, the {answer} has growing features: containers, soil-holding spaces, or structures specifically designed for planting and cultivating vegetation. Other objects lack plant cultivation capabilities."
            }
        }
        
        # Property-based templates
        self.property_templates = {
            'flammability': {
                'flammable_materials': ['paper', 'wood', 'fabric', 'textiles', 'cotton', 'certain plastics'],
                'non_flammable_materials': ['metal', 'glass', 'stone', 'ceramic'],
                'reasoning_template': "Flammable materials include {flammable_materials} while non-flammable materials include {non_flammable_materials}. Among {objects}, the {answer} is made of flammable materials while others are non-flammable."
            },
            'thermal_touch': {
                'warm_materials': ['wood', 'fabric', 'plastic'],
                'cool_materials': ['metal', 'glass', 'stone'],
                'reasoning_template': "Materials vary in thermal properties with {warm_materials} feeling warm and {cool_materials} feeling cool. Among {objects}, the {answer} has the expected thermal properties for its material type."
            },
            'sound_absorption': {
                'absorbent_materials': ['fabric', 'foam', 'carpet', 'soft materials'],
                'reflective_materials': ['metal', 'glass', 'hard surfaces'],
                'reasoning_template': "Sound-absorbing materials include {absorbent_materials} while sound-reflecting materials include {reflective_materials}. Among {objects}, the {answer} has sound absorption properties while others reflect sound."
            },
            'scratch_resistance': {
                'resistant_materials': ['metal', 'glass', 'ceramic', 'hard plastics'],
                'soft_materials': ['wood', 'fabric', 'soft plastics'],
                'reasoning_template': "Scratch-resistant materials include {resistant_materials} while softer materials include {soft_materials}. Among {objects}, the {answer} has scratch resistance while others are more susceptible to scratching."
            },
            'latent_compressible': {
                'compressible_materials': ['fabric', 'foam', 'sponge', 'soft materials'],
                'rigid_materials': ['metal', 'glass', 'stone', 'hard plastics'],
                'reasoning_template': "Compressible materials include {compressible_materials} while rigid materials include {rigid_materials}. Among {objects}, the {answer} has compressible properties while others are rigid."
            },
            'light': {
                'visual_clues': ['slim or airy frame', 'thin panels or fabric', 'minimal bulk'],
                'positive_traits': ['lightweight construction', 'easy to lift or reposition', 'minimal structural mass'],
                'negative_traits': ['dense framing', 'solid heavy core', 'requires noticeable effort to move'],
                'reasoning_template': "Lightweight materials show {visual_clues} and {positive_traits}. Among {objects}, the {answer} appears lightweight while other options look denser or harder to move."
            },
            'heavy': {
                'visual_clues': ['thick supports', 'solid bulky silhouette', 'dense structural elements'],
                'positive_traits': ['substantial weight', 'solid dense build', 'resists being moved easily'],
                'negative_traits': ['slim lightweight frame', 'minimal structural mass', 'easy to lift construction'],
                'reasoning_template': "Heavy objects display {visual_clues} and {positive_traits}. Among {objects}, the {answer} has heavy characteristics while alternatives appear noticeably lighter."
            },
            
        }
        
        # Function templates
        self.function_templates = {
            'display_exhibit': "Display objects are designed for visibility and presentation. Among {objects}, the {answer} is designed for display purposes while others serve different functions.",
            'storage_organization': "Storage objects provide space for keeping items organized. Among {objects}, the {answer} is designed for storage while others lack storage capabilities.",
            'food_preparation': "Food preparation objects are designed for cooking and food handling. Among {objects}, the {answer} is designed for food preparation while others serve different purposes.",
            'cleaning_sanitation': "Cleaning objects are designed for hygiene and maintenance. Among {objects}, the {answer} is designed for cleaning while others serve different functions.",
            'transportation': "Transportation objects are designed for moving people or goods. Among {objects}, the {answer} is designed for transportation while others are stationary.",
            'communication': "Communication objects facilitate information exchange. Among {objects}, the {answer} is designed for communication while others serve different purposes.",
            'entertainment': "Entertainment objects provide amusement and recreation. Among {objects}, the {answer} is designed for entertainment while others serve practical functions.",
            'decoration': "Decorative objects enhance visual appeal and aesthetics. Among {objects}, the {answer} is designed for decoration while others serve functional purposes.",
            'lighting': "Lighting objects provide illumination and visibility. Among {objects}, the {answer} is designed for lighting while others serve different functions.",
            'measurement': "Measurement objects are designed for quantifying dimensions or quantities. Among {objects}, the {answer} is designed for measurement while others serve different purposes.",
            'knowledge': "To answer this question, I examine the knowledge and information capabilities of each object. Among {objects}, the {answer} relates to knowledge functions: storing, displaying, or providing information for learning and reference. Other objects lack knowledge or information-related functions.",
            'seating': "Seating objects provide comfortable surfaces for sitting with appropriate height, support, and ergonomics. Among {objects}, the {answer} is designed specifically for seating purposes with visible seating characteristics. Other objects lack the structural and dimensional requirements for comfortable seating.",
            'foldable': "Foldable objects can be collapsed or folded to save space, made of flexible materials that allow transformation. Among {objects}, the {answer} shows foldable characteristics: materials and structure that enable folding, collapsing, or compact storage. Other objects are rigid and cannot be folded."
        }
        
        # Repurposing templates
        self.repurposing_templates = {
            'container_concept': "Container repurposing requires hollow space and accessibility. Among {objects}, the {answer} has suitable characteristics for container use while others lack these features.",
            'reflector_concept': "Reflector repurposing requires smooth surface and reflective properties. Among {objects}, the {answer} has suitable characteristics for reflection while others lack these features.",
            'cushion_concept': "Cushion repurposing requires soft materials and compressible properties. Among {objects}, the {answer} has suitable characteristics for cushioning while others are too rigid.",
            'stepstool_concept': "Step stool repurposing requires stable structure and appropriate height. Among {objects}, the {answer} has suitable characteristics for stepping while others lack stability.",
            'bookend_concept': "Bookend repurposing requires weight and appropriate shape. Among {objects}, the {answer} has suitable characteristics for holding books while others lack the necessary weight or shape.",
            'tool_concept': "Tool repurposing requires appropriate shape and material properties. Among {objects}, the {answer} has suitable characteristics for tool use while others lack these features.",
            'shield_concept': (
                "Shield repurposing demands a broad, rigid surface that is durable enough to absorb impact, "
                "large enough to cover the body, and still movable so it can be repositioned quickly. "
                "Among {objects}, the {answer} offers that protective, sturdy coverage while remaining movable, "
                "whereas the other options cannot serve as reliable, repositionable shields."
            )
        }

        self.qa_space_descriptions: Dict[str, str] = {}
        qa_space_root = Path(__file__).resolve().parents[2]
        qa_space_path = qa_space_root / "analysis/results/question_answer_space_analysis.json"
        if qa_space_path.exists():
            try:
                qa_space_data = json.loads(qa_space_path.read_text())
                mappings = qa_space_data.get("question_answer_mappings", qa_space_data)
                if isinstance(mappings, dict):
                    self.qa_space_descriptions = {
                        self._normalize_cluster_key(key): value.get("description", "")
                        for key, value in mappings.items()
                        if isinstance(value, dict) and value.get("description")
                    }
            except Exception:  # pragma: no cover - defensive
                logger.warning("Failed to load QA-space descriptions for reasoning enrichment", exc_info=True)
                self.qa_space_descriptions = {}

        self._build_cluster_indexes()
        
        # Spatial reasoning templates emphasising anchor orientation
        self.spatial_templates = {
            'left_right': (
                "I anchor on {reference_phrase} and inspect its orientation cues to determine which way its front faces. "
                "With that front established, I compare horizontal positions. {relation_clause} "
                "Therefore {answer_label} is correct while the other option does not occupy that side relative to the anchor."
            ),
            'front_behind': (
                "I anchor on {reference_phrase} and establish its front-facing direction. "
                "Using depth and occlusion cues, I evaluate where the compared object sits along that front-to-back axis. {relation_clause} "
                "This confirms {answer_label} as the correct choice while other candidates lie on the opposite side."
            ),
            'above_below': (
                "I anchor on {reference_phrase} and establish a vertical frame of reference using its geometry. "
                "Comparing heights reveals where the other object sits vertically. {relation_clause} "
                "As a result {answer_label} is correct while alternatives are positioned differently."
            ),
            'closer_to_camera': (
                "I anchor on {reference_phrase} and establish its facing direction, then analyze scale, overlap, and perspective cues to judge depth. "
                "{relation_clause} "
                "Thus {answer_label} is correct because other objects appear at different depths."
            )
        }

    def clean_text(self, text: str) -> str:
        return core.clean_text(text)
    
    def _normalize_cluster_key(self, text: Optional[str]) -> str:
        return core.normalize_cluster_key(text)

    def _template_to_summary(self, template: Any) -> Optional[str]:
        if isinstance(template, str):
            summary = template.replace('{objects}', 'the objects').replace('{answer}', 'the object')
            return self.clean_text(summary)
        return None

    def _description_to_feature_clause(self, description: str) -> str:
        if not description:
            return ""
        desc = description.strip()
        if '(' in desc and ')' in desc:
            start = desc.find('(')
            end = desc.find(')', start + 1)
            if start != -1 and end != -1:
                inner = desc[start + 1:end].strip()
                if inner:
                    desc = inner
        desc = desc.replace('/', ' or ')
        desc = desc.replace('>', ' greater than ')
        cleaned = self.clean_description_punctuation(desc)
        cleaned = cleaned.replace(' ,', ', ')
        cleaned = re.sub(r'\s+or\s+', ' or ', cleaned)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        cleaned = cleaned.replace(
            'not food or electronics or furniture or large architectural items',
            'not food, electronics, furniture, or large architectural items'
        )
        return cleaned.strip().rstrip('.')

    def _build_cluster_indexes(self) -> None:
        self._material_feature_index: Dict[str, Dict[str, Any]] = {
            self._normalize_cluster_key(key): value for key, value in self.material_templates.items()
        }
        self._affordance_feature_index: Dict[str, Dict[str, Any]] = {
            self._normalize_cluster_key(key): value for key, value in self.affordance_templates.items()
        }
        self._function_feature_index: Dict[str, str] = {}
        for key, value in self.function_templates.items():
            summary = self._template_to_summary(value)
            if summary:
                self._function_feature_index[self._normalize_cluster_key(key)] = summary
        self._property_feature_index: Dict[str, Dict[str, Any]] = {
            self._normalize_cluster_key(key): value for key, value in self.property_templates.items()
        }
        self._repurposing_feature_index: Dict[str, Any] = {
            self._normalize_cluster_key(key): value for key, value in self.repurposing_templates.items()
        }

    def _describe_cluster_features(self, filter_type: str, cluster_name: Optional[str]) -> Optional[str]:
        override = core.cluster_feature_override(cluster_name or "")
        if override:
            return override
        normalized = self._normalize_cluster_key(cluster_name)
        if not normalized:
            return None
        category = (filter_type or "").lower().replace(" ", "_")
        summary_parts: List[str] = []

        if category == "affordance":
            data = self._affordance_feature_index.get(normalized)
            if isinstance(data, dict):
                features = ', '.join(data.get('features', [])[:2])
                requirements = ', '.join(data.get('requirements', [])[:2])
                if features:
                    summary_parts.append(f"features like {features}")
                if requirements:
                    summary_parts.append(f"supporting {requirements}")
        elif category == "function":
            summary = self._function_feature_index.get(normalized)
            if summary:
                summary_parts.append(summary)
        elif category == "material":
            data = self._material_feature_index.get(normalized)
            if isinstance(data, dict):
                visual = ', '.join(data.get('visual', [])[:2])
                physical = ', '.join(data.get('physical', [])[:2])
                if visual:
                    summary_parts.append(f"visual cues such as {visual}")
                if physical:
                    summary_parts.append(f"physical traits like {physical}")
        elif category == "physical_property":
            data = self._property_feature_index.get(normalized)
            if isinstance(data, dict):
                visual_list = data.get("visual_clues") or data.get("visual_traits") or []
                positive_list = (
                    data.get("positive_traits")
                    or data.get("traits")
                    or data.get("flammable_materials")
                    or data.get("warm_materials")
                    or data.get("absorbent_materials")
                    or data.get("resistant_materials")
                    or data.get("compressible_materials")
                    or []
                )
                negative_list = (
                    data.get("negative_traits")
                    or data.get("contrasts")
                    or data.get("non_flammable_materials")
                    or data.get("cool_materials")
                    or data.get("reflective_materials")
                    or data.get("soft_materials")
                    or data.get("rigid_materials")
                    or []
                )
                if visual_list:
                    summary_parts.append(f"visual cues such as {', '.join(visual_list[:2])}")
                if positive_list:
                    summary_parts.append(f"traits such as {', '.join(positive_list[:2])}")
                if negative_list:
                    summary_parts.append(f"contrasting with {', '.join(negative_list[:2])}")
                if not summary_parts:
                    positive = None
                    negative = None
                    for key, values in data.items():
                        if not isinstance(values, list):
                            continue
                        joined = ', '.join(values[:2])
                        if not joined:
                            continue
                        if key.startswith(('flammable', 'warm', 'absorbent', 'resistant', 'compressible', 'light', 'heavy')):
                            if not positive:
                                positive = joined
                        elif key.startswith(('non_flammable', 'cool', 'reflective', 'soft', 'rigid')):
                            if not negative:
                                negative = joined
                    if positive:
                        summary_parts.append(f"traits such as {positive}")
                    if negative:
                        summary_parts.append(f"contrasting with {negative}")

        if not summary_parts:
            description = self.qa_space_descriptions.get(normalized)
            if description:
                summary_parts.append(self._description_to_feature_clause(description))

        combined = self._summarize_feature_list(summary_parts)
        return combined

    def _summarize_feature_list(self, clauses: List[str]) -> Optional[str]:
        return core.summarize_feature_list(clauses)

    def _collect_material_entries(self, obj: str) -> List[Dict[str, Any]]:
        """Gather material clusters with descriptive cues for an object."""
        entries: List[Dict[str, Any]] = []
        if not self.taxonomy_utils:
            return entries
        try:
            clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_material') or []
        except Exception:
            clusters = []
        for cluster in clusters:
            if not cluster:
                continue
            key = self._normalize_cluster_key(cluster)
            template = self._material_feature_index.get(key, {})
            visual = template.get("visual") or []
            physical = template.get("physical") or []
            label = self._naturalize_cluster_phrase(cluster, "material") or self.clean_cluster_name(cluster)
            if not label and not (visual or physical):
                continue
            entries.append(
                {
                    "label": label or cluster,
                    "visual": visual,
                    "physical": physical,
                }
            )
        return entries

    def _describe_requirement_trait(self, name: str) -> Optional[str]:
        """Return descriptive text for a taxonomy requirement."""
        if not name:
            return None
        for domain in ("physical_property", "affordance", "function", "material"):
            description = self._describe_cluster_features(domain, name)
            if description:
                return description
        return None

    def _format_object_list(self, items: List[str]) -> str:
        return core.format_object_list(items)

    def _render_domain_reference(self, domain: Optional[str], label: Optional[str]) -> Optional[str]:
        return core.render_domain_reference(domain, label)

    def _compose_entry_summary(self, entry: Dict[str, Any]) -> Optional[str]:
        return core.compose_entry_summary(entry)

    def _render_filter_trait(
        self,
        filter_type: Optional[str],
        filter_value: Optional[str],
        feature_summary: Optional[str],
    ) -> str:
        return core.render_filter_trait(filter_type, filter_value, feature_summary)

    def _invert_box_mapping(self, box_to_object: Optional[Dict[str, str]]) -> Dict[str, List[str]]:
        """Convert a box→object mapping into object→box labels."""
        inverted: Dict[str, List[str]] = {}
        if isinstance(box_to_object, dict):
            for label, obj in box_to_object.items():
                if not obj:
                    continue
                inverted.setdefault(obj, []).append(label)
        return inverted

    def build_object_recognition_steps(
        self,
        object_set: List[str],
        box_to_object: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Create labeled steps that describe bounding boxes and their objects."""
        steps: List[Dict[str, Any]] = []
        inverted = self._invert_box_mapping(box_to_object)
        for obj in object_set:
            labels = inverted.get(obj)
            if not labels:
                norm_obj = self._normalize_object_name(obj)
                for candidate, candidate_labels in inverted.items():
                    if self._normalize_object_name(candidate) == norm_obj:
                        labels = candidate_labels
                        break
            if labels:
                label_text = " / ".join(self._normalize_box_label(label) for label in labels)
                description = f"{label_text} corresponds to {obj}".strip()
                steps.append(
                    self._create_reasoning_step(
                        "object_recognition",
                        description,
                        input_data=label_text,
                        output_data=obj,
                    )
                )
            else:
                description = f"No color-box annotation corresponds to {obj}; object identified from context"
                steps.append(
                    self._create_reasoning_step(
                        "object_recognition",
                        description,
                        output_data=obj,
                    )
                )
        return steps

    @staticmethod
    def _normalize_box_label(label: str) -> str:
        """Normalize box label text to emphasize the color-box mapping."""
        if not label:
            return ""
        normalized = label.replace("_", " ").strip()
        # Enforce consistent phrasing ending with 'box' where applicable.
        if "box" not in normalized.lower():
            normalized = f"{normalized} box"
        return normalized

    @staticmethod
    def _normalize_object_name(name: Optional[str]) -> str:
        if not name:
            return ""
        return str(name).strip().lower()

    def _format_answer_text(
        self,
        answer: str,
        box_to_object: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create the answer step with optional bounding box reference."""
        if not answer:
            return "unspecified"
        inverted = self._invert_box_mapping(box_to_object)
        labels = inverted.get(answer)
        if labels:
            label_text = " or ".join(labels)
            return f"{answer} ({label_text})"
        return str(answer)

    def get_reasoning_label(self, question_type: str) -> str:
        """Map question types to structured reasoning labels."""
        if question_type.startswith("spatial_"):
            return "Spatial analysis"
        if question_type.startswith("material_") or question_type == "material_property":
            return "Material analysis"
        if question_type.startswith("physical_"):
            return "Physical analysis"
        if question_type.startswith("affordance_"):
            return "Affordance analysis"
        if question_type.startswith("function_") or question_type.startswith("functional_"):
            return "Functional analysis"
        if question_type.startswith("repurposing_"):
            return "Repurposing analysis"
        if question_type.startswith("compositional_"):
            return "Compositional analysis"
        if question_type.startswith("description_"):
            return "Description matching"
        if question_type.startswith("counterfactual_"):
            return "Counterfactual analysis"
        if question_type.startswith("latent_"):
            return "Latent property analysis"
        return "Reasoning"

    def format_structured_reasoning(
        self,
        steps: List[Tuple[str, str]],
    ) -> str:
        """Format reasoning steps into a numbered, multi-line narrative."""
        if not steps:
            return ""
        lines = ["Reasoning"]
        for idx, (label, text) in enumerate(steps, start=1):
            clean_label = label.strip()
            cleaned = self.clean_description_punctuation(text or "")
            if cleaned and clean_label.lower() != "<answer>" and cleaned[-1] not in (".", "!", "?", ";"):
                cleaned = f"{cleaned}."
            lines.append(f"{idx} {clean_label} {cleaned}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_reasoning_value(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, (list, tuple)):
            return {"items": list(value)}
        return {"value": value}

    def _create_reasoning_step(
        self,
        reasoning_type: str,
        description: Optional[str] = None,
        *,
        steps: Optional[List[Dict[str, Any]]] = None,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_type = self.STEP_TYPE_NORMALIZATION.get(reasoning_type, reasoning_type or "analysis")
        step: Dict[str, Any] = {
            "reasoning_type": normalized_type,
        }
        if description:
            step["description"] = description
        if steps:
            step["steps"] = steps
        inputs_normalized = self._normalize_reasoning_value(input_data)
        if inputs_normalized:
            step["inputs"] = inputs_normalized
        outputs_normalized = self._normalize_reasoning_value(output_data)
        if outputs_normalized:
            step["outputs"] = outputs_normalized
        if extra:
            if isinstance(extra, dict):
                step.setdefault("metadata", {}).update(extra)
            else:
                step.setdefault("metadata", {})["value"] = extra
        return step

    def render_reasoning_text(self, structured_steps: List[Dict[str, Any]]) -> str:
        """Render phase-based reasoning into a numbered narrative."""
        if not structured_steps:
            return ""
        lines = ["Reasoning"]
        for idx, phase in enumerate(structured_steps, start=1):
            phase_name = phase.get("phase", "analysis")
            steps = phase.get("steps") or []
            summary = phase.get("summary")
            if steps:
                for step_idx, sentence in enumerate(steps, start=1):
                    cleaned = self.clean_text(sentence)
                    if cleaned and cleaned[-1] not in ".!?;":
                        cleaned = f"{cleaned}."
                    lines.append(f"{idx}.{step_idx} <{phase_name}> {cleaned}")
            if summary:
                cleaned_summary = self.clean_text(summary)
                if cleaned_summary and cleaned_summary[-1] not in ".!?;":
                    cleaned_summary = f"{cleaned_summary}."
                lines.append(f"{idx} <{phase_name} summary> {cleaned_summary}")
            if phase_name == "answer":
                final_answer = phase.get("final_answer")
                rationale = phase.get("rationale")
                answer_line = final_answer
                if rationale:
                    answer_line = f"{final_answer} — {self.clean_text(rationale)}"
                if answer_line and answer_line[-1] not in ".!?;":
                    answer_line = f"{answer_line}."
                lines.append(f"{idx} <answer> {answer_line}")
        return "\n".join(lines)

    def compose_structured_reasoning(
        self,
        question_type: str,
        object_set: List[str],
        answer: str,
        core_steps: List[Any],
        box_to_object: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Compile structured reasoning steps for downstream use."""
        structured_steps: List[Dict[str, Any]] = []
        recognition_steps = self.build_object_recognition_steps(object_set, box_to_object)
        if recognition_steps:
            structured_steps.extend(recognition_steps)
        if core_steps:
            structured_steps.extend(core_steps)
        answer_text = self._format_answer_text(answer, box_to_object)
        structured_steps.append(
            self._create_reasoning_step(
                "answer",
                f"Final answer: {answer_text}",
            )
        )
        return structured_steps
    
    def clean_cluster_name(self, cluster_name: str) -> str:
        return core.clean_cluster_name(cluster_name)

    def _cluster_label(self, domain: Optional[str], plural: bool = False) -> str:
        return core.cluster_label(domain, plural)

    def _naturalize_cluster_phrase(self, cluster_name: Optional[str], domain: Optional[str] = None) -> Optional[str]:
        return core.naturalize_cluster_phrase(cluster_name, domain)
    
    def clean_description_punctuation(self, text: str) -> str:
        return core.clean_description_punctuation(text)

    def generate_material_reasoning(self, question_type: str, target_object: str, object_set: List[str], answer: str, **kwargs) -> List[Dict[str, Any]]:
        """Generate structured material reasoning steps."""
        material_key = question_type.replace('material_', '')
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
            others_text = ', '.join(others)
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
                self._create_reasoning_step(
                    "analysis",
                    f"I focus on objects already noted for {qa_filter_desc}.",
                )
            )

        template_data = None
        for key, data in self.material_templates.items():
            if key in material_key or any(word in material_key for word in key.split('_') if len(word) > 3):
                template_data = data
                break
        if not template_data:
            if any(token in material_key for token in ['textiles', 'fibers', 'leather']):
                template_data = self.material_templates.get('textiles_fibers_and_leather')
            elif 'wood' in material_key or 'plant' in material_key:
                template_data = self.material_templates.get('wood_and_plant_based_solids')
            elif 'metal' in material_key or 'alloy' in material_key:
                template_data = self.material_templates.get('metals_and_alloys')

        material_name = kwargs.get('material', material_key.replace('_', ' ').strip())
        answer_entries = self._collect_material_entries(answer)
        
        if template_data:
            visual_props = ', '.join(template_data['visual'][:2])
            physical_props = ', '.join(template_data['physical'][:2])
            visual_summary = visual_props or None
            physical_summary = physical_props or None
            steps.append(
                self._create_reasoning_step(
                    "material_properties",
                    f"The {answer} shows material cues such as {visual_props} and feels {physical_props}.",
                    output_data=answer,
                    extra={"visual_properties": visual_props, "physical_properties": physical_props},
                )
            )
            positive_summary = visual_summary or physical_summary or positive_summary
        elif answer_entries:
            entry = answer_entries[0]
            visual_summary = ', '.join(entry.get("visual", [])[:2]) or None
            physical_summary = ', '.join(entry.get("physical", [])[:2]) or None
            descriptor_bits: List[str] = []
            if visual_summary:
                descriptor_bits.append(f"material cues such as {visual_summary}")
            if physical_summary:
                descriptor_bits.append(f"feels {physical_summary}")
            if descriptor_bits:
                steps.append(
                    self._create_reasoning_step(
                        "material_properties",
                        f"The {answer} shows {self.clean_text(' and '.join(descriptor_bits))}.",
                        output_data=answer,
                    )
            )
        else:
            steps.append(
                self._create_reasoning_step(
                    "material_properties",
                    f"I examine material properties of each object to see which matches {material_name}.",
                )
            )

        if visual_summary and not positive_summary:
            positive_summary = visual_summary
        if physical_summary and not positive_summary:
            positive_summary = physical_summary

        if any(prop in material_key for prop in ['sound_absorption', 'flammability', 'thermal_touch', 'scratch_resistance', 'latent_compressible', 'electrical_conductivity']):
            prop_template = self._get_property_template_for_material_question(material_key)
            if prop_template:
                absorb_props = {k: ', '.join(v[:2]) for k, v in prop_template.items() if isinstance(v, list)}
                if 'thermal_touch' in material_key:
                    positive_summary = absorb_props.get('cool_materials')
                    negative_summary = absorb_props.get('warm_materials')
                elif 'sound_absorption' in material_key:
                    positive_summary = absorb_props.get('absorbent_materials')
                    negative_summary = absorb_props.get('reflective_materials')
                elif 'scratch_resistance' in material_key:
                    positive_summary = absorb_props.get('resistant_materials')
                    negative_summary = absorb_props.get('soft_materials')
                elif 'latent_compressible' in material_key:
                    positive_summary = absorb_props.get('compressible_materials')
                    negative_summary = absorb_props.get('rigid_materials')
                elif 'flammability' in material_key:
                    positive_summary = absorb_props.get('flammable_materials')
                    negative_summary = absorb_props.get('non_flammable_materials')
                elif 'electrical_conductivity' in material_key:
                    positive_summary = absorb_props.get('conductive_materials')
                    negative_summary = absorb_props.get('insulating_materials')
                positive_summary = positive_summary or absorb_props.get('absorbent_materials') or absorb_props.get('resistant_materials') or absorb_props.get('warm_materials')
                negative_summary = negative_summary or absorb_props.get('reflective_materials') or absorb_props.get('soft_materials') or absorb_props.get('cool_materials')

        elimination_sentences: List[str] = []
        for obj in others:
            object_entries = self._collect_material_entries(obj)
            if object_entries:
                entry = object_entries[0]
                bits: List[str] = []
                if entry.get("visual"):
                    bits.append(f"visual cues such as {', '.join(entry['visual'][:2])}")
                if entry.get("physical"):
                    bits.append(f"traits like {', '.join(entry['physical'][:2])}")
                detail = self._summarize_feature_list(bits) if bits else None
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
                self._create_reasoning_step(
                    "elimination",
                    "Other choices are ruled out because " + " ".join(elimination_sentences),
                    input_data={"objects": others},
                )
            )

        if elimination_feature_details:
            elimination_summary = self._summarize_feature_list(elimination_feature_details)
            if elimination_summary and not negative_summary:
                negative_summary = elimination_summary

        steps.append(
            self._create_reasoning_step(
                "thinking",
                self._compose_material_thinking_sentence(
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

    def _compose_material_thinking_sentence(
        self,
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
            contrast_clause = (
                f", while {others_clause} {'do' if others_verb == 'are' else 'does'} relate to {negative_summary}"
            )
        else:
            contrast_clause = f", whereas {others_clause} {'do' if others_verb == 'are' else 'does'} not."
        sentence += contrast_clause
        if not sentence.endswith("."):
            sentence += "."
        return self.clean_text(sentence)

    def _compose_physical_property_analysis_sentence(
        self,
        property_name: str,
        visual_clues: List[str],
        positive_traits: List[str],
    ) -> str:
        fragments: List[str] = []
        if visual_clues:
            fragments.append(f"visual cues such as {', '.join(visual_clues[:2])}")
        if positive_traits:
            fragments.append(f"traits like {', '.join(positive_traits[:2])}")
        if fragments:
            if len(fragments) == 1:
                descriptor = fragments[0]
            else:
                descriptor = ", ".join(fragments[:-1]) + f", and {fragments[-1]}"
            return f"I compare {descriptor} for each object to see who shows {property_name.lower()}."
        return f"I compare form and visible cues of each object to see who shows {property_name}."

    def _compose_physical_property_thinking_sentence(
        self,
        answer: str,
        property_name: str,
        others_clause: str,
        others_verb: str,
        visual_clues: List[str],
        positive_traits: List[str],
    ) -> str:
        feature_bits: List[str] = []
        if visual_clues:
            feature_bits.append(f"visual cues such as {', '.join(visual_clues[:2])}")
        if positive_traits:
            feature_bits.append(f"traits like {', '.join(positive_traits[:2])}")
        feature_summary = self._summarize_feature_list(feature_bits)
        property_label = property_name.lower()
        if feature_summary:
            return (
                f"The {answer} exhibits {feature_summary}, so it best demonstrates {property_label} while "
                f"{others_clause} {others_verb} not."
            )
        return f"Therefore the {answer} best exhibits {property_label} while {others_clause} {others_verb} not."
    
    def _get_property_template_for_material_question(self, material_key: str) -> Optional[Dict]:
        """Get property template for material questions that don't match material templates"""
        # Map material question types to property templates
        property_mapping = {
            'sound_absorption': 'sound_absorption',
            'flammability': 'flammability',
            'thermal_touch': 'thermal_touch',
            'scratch_resistance': 'scratch_resistance',
            'latent_compressible': 'latent_compressible',
            'electrical_conductivity': 'electrical_conductivity'
        }
        
        for key, prop_type in property_mapping.items():
            if key in material_key:
                return self.property_templates.get(prop_type)
        
        return None
    
    def _find_template_match(self, key: str, templates: Dict, min_word_length: int = 3) -> Any:
        """Helper method to find template matches with consistent logic"""
        # Try exact match first
        if key in templates:
            return templates[key]
        
        # Try partial match with word filtering
        for template_key, template_data in templates.items():
            if any(word in key for word in template_key.split('_') if len(word) > min_word_length):
                return template_data
        
        return None

    def generate_affordance_reasoning(
        self,
        question_type: str,
        target_object: str,
        object_set: List[str],
        answer: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate structured affordance reasoning for any affordance type."""
        affordance_key = question_type.replace('affordance_', '')
        
        # Normalize the key for matching
        normalized_key = affordance_key.replace('__', '_').replace('_', '_')
        
        # Try to find matching affordance type with multiple attempts
        template_data = None
        
        # Try direct match
        if normalized_key in self.affordance_templates:
            template_data = self.affordance_templates[normalized_key]
        
        # Try finding by partial match
        if not template_data:
            for key, data in self.affordance_templates.items():
                # Remove special characters for comparison
                clean_key = key.replace('(', '').replace(')', '').replace('_', '').replace('—', '').replace(',', '').replace(' ', '').lower()
                clean_affordance = normalized_key.replace('(', '').replace(')', '').replace('_', '').replace('—', '').replace(',', '').replace(' ', '').lower()

                if clean_affordance in clean_key or clean_key in clean_affordance:
                template_data = data
                break
        
        if template_data:
            features = ', '.join(template_data['features'][:2])
            requirements = ', '.join(template_data['requirements'][:2])
            steps = [
                self._create_reasoning_step(
                    "affordance_analysis",
                    f"The {answer} offers features such as {features} and meets requirements like {requirements}"
                ),
                self._create_reasoning_step(
                    "thinking",
                    f"These affordances make the {answer} suitable while alternatives fail to meet all requirements"
                ),
            ]
        else:
            steps = self._generate_taxonomy_enhanced_reasoning(
                question_type, target_object, object_set, answer, "affordance"
            )
        
        return steps

    def _build_spatial_relation_clause(
        self,
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
        
        if question_type == 'spatial_left_right':
            if relation_value in {'left', 'right'}:
                clause = f"{target_phrase} sits on the {relation_value} side of {reference_phrase}."
            elif answer_term in {'left', 'right'}:
                clause = f"{target_phrase} sits on the {answer_term} side of {reference_phrase}."
        else:
                clause = f"{target_phrase} occupies the side of {reference_phrase} described in the question, while the other option falls on the opposite side."
        elif question_type == 'spatial_front_behind':
            if relation_value == 'front':
                clause = f"{target_phrase} stands in front of {reference_phrase} along that facing direction."
            elif relation_value == 'behind':
                clause = f"{target_phrase} sits behind {reference_phrase} once the front is established."
            elif answer_term in {'front', 'behind'}:
                clause = f"{target_phrase} aligns on the {answer_term} side of {reference_phrase}, unlike the other option."
            else:
                clause = f"{target_phrase} aligns with the front-or-behind relationship described for {reference_phrase}, unlike the alternatives."
        elif question_type == 'spatial_above_below':
            if relation_value in {'above', 'below'}:
                clause = f"{target_phrase} is {relation_value} {reference_phrase}, matching the vertical arrangement."
            elif answer_term in {'above', 'below'}:
                clause = f"{target_phrase} is {answer_term} {reference_phrase}, matching the vertical arrangement."
            else:
                clause = f"{target_phrase} occupies the correct vertical position relative to {reference_phrase}, whereas others do not."
        elif question_type == 'spatial_closer_to_camera':
            if relation_value == 'closer':
                clause = f"{target_phrase} appears closer to the camera than {reference_phrase} based on scale and occlusion cues."
            elif relation_value == 'farther':
                clause = f"{target_phrase} appears farther from the camera than {reference_phrase} when perspective cues are evaluated."
            elif answer_term in {'closer', 'farther'}:
                clause = f"{target_phrase} appears {answer_term} relative to {reference_phrase} when perspective cues are evaluated."
            else:
                clause = f"{target_phrase} matches the depth relationship described relative to {reference_phrase}."
        else:
            clause = f"{target_phrase} matches the spatial relationship to {reference_phrase} described in the question."
        
        if clause and clause[0].islower():
            clause = clause[0].upper() + clause[1:]
        return clause

    def generate_function_reasoning(
        self,
        question_type: str,
        target_object: str,
        object_set: List[str],
        answer: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate structured function reasoning for any function type."""
        if question_type in ('function_seating', 'functional_seating'):
            # Reuse the seating affordance reasoning template for functional seating
            return self.generate_affordance_reasoning('affordance_sit__ride__attend', target_object, object_set, answer)

        function_key = question_type.replace('function_', '').replace('_', '_')
        
        # Try to find matching function type
        template = self._find_template_match(function_key, self.function_templates)
        
        if template:
            objects_str = ', '.join(object_set)
            # Only include keys that exist in the template to avoid KeyError
            import string
            template_placeholders = [field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name]
            format_kwargs = {k: v for k, v in {'objects': objects_str, 'answer': answer}.items() if k in template_placeholders}
            analysis_sentence = self.clean_text(template.format(**format_kwargs))
            steps = [
                self._create_reasoning_step("functional_analysis", analysis_sentence),
                self._create_reasoning_step(
                    "thinking",
                    f"The {answer} fulfills the required function while other objects do not"
                ),
            ]
        else:
            # Enhanced generic function reasoning with detailed elimination analysis
            steps = self._generate_taxonomy_enhanced_reasoning(
                question_type, target_object, object_set, answer, "function"
            )
        
        return steps
    
    def generate_physical_property_reasoning(
        self,
        question_type: str,
        target_object: str,
        object_set: List[str],
        answer: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate structured reasoning for physical property questions."""
        property_name = kwargs.get('physical_property')
        cleaned_clusters: List[str] = []
        others = [obj for obj in object_set if obj != answer]
        if not others:
            others_clause = "the other objects"
            others_verb = "do"
        elif len(others) == 1:
            others_clause = others[0]
            others_verb = "does"
        else:
            others_clause = ', '.join(others)
            others_verb = "do"
        property_key = None
        visual_clues: List[str] = []
        positive_traits: List[str] = []
        negative_traits: List[str] = []
        
        if self.taxonomy_utils:
            try:
                clusters = self.taxonomy_utils.get_object_clusters(answer, 'final_taxonomy_physical_properties') or []
                cleaned_clusters = [
                    self.clean_cluster_name(cluster)
                    for cluster in clusters
                    if cluster and not is_void_cluster(cluster, 'physical')
                ]
            except Exception:
                cleaned_clusters = []
        
        if not property_name and cleaned_clusters:
            property_name = cleaned_clusters[0]
        
        if not property_name:
            property_name = "the specified physical property"

        property_key = self._normalize_cluster_key(property_name)
        property_template = self._property_feature_index.get(property_key) if property_key else None
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
                self._create_reasoning_step(
                    "analysis",
                    f"I focus on objects already known for {qa_filter_desc}."
                )
            )

        steps.append(
            self._create_reasoning_step(
                "physical_properties",
                self.clean_text(
                    self._compose_physical_property_analysis_sentence(
                        property_name,
                        visual_clues,
                        positive_traits,
                    )
                )
            )
        )

        if cleaned_clusters:
            group_phrases = [
                self._render_domain_reference("physical_property", name)
                for name in cleaned_clusters[:2]
                if name
            ]
            group_summary = self._summarize_feature_list(group_phrases)
            if group_summary:
                steps.append(
                    self._create_reasoning_step(
                        "physical_properties",
                        f"The {answer} consistently shows {group_summary}.",
                    )
                )
        elif visual_clues or positive_traits:
            descriptive_bits: List[str] = []
            if visual_clues:
                descriptive_bits.append(f"visual cues such as {', '.join(visual_clues[:2])}")
            if positive_traits:
                descriptive_bits.append(f"traits like {', '.join(positive_traits[:2])}")
            if descriptive_bits:
                steps.append(
                    self._create_reasoning_step(
                        "physical_properties",
                        f"The {answer} shows {self.clean_text(', '.join(descriptive_bits))}."
                    )
                )
        
        elimination_details: List[str] = []
        if self.taxonomy_utils:
            for obj in object_set:
                if obj == answer:
                    continue
                try:
                    other_clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_physical_properties') or []
                    filtered_clusters = [
                        cluster for cluster in other_clusters
                        if cluster and not is_void_cluster(cluster, 'physical')
                    ]
                    natural_names: List[str] = []
                    feature_bits: List[str] = []
                    for cluster in filtered_clusters:
                        natural_name = self._naturalize_cluster_phrase(cluster, "physical") or self.clean_cluster_name(cluster)
                        if natural_name:
                            natural_names.append(natural_name)
                        described = self._describe_cluster_features('physical_property', cluster)
                        if described:
                            feature_bits.append(described)
                    detail = self._summarize_feature_list(feature_bits)
                    if detail:
                        elimination_details.append(f"{obj} shows {detail}, which does not express {property_name.lower()}.")
                    elif natural_names:
                        elimination_details.append(f"{obj} aligns with {', '.join(natural_names[:2])}, which does not express {property_name.lower()}.")
        else:
                        elimination_details.append(f"{obj} offers no physical traits that align with {property_name.lower()}.")
                except Exception:
                    elimination_details.append(f"{obj} does not present evidence of {property_name}.")
        else:
            if negative_traits:
                elimination_details.append(
                    f"{others_clause} {others_verb} show {', '.join(negative_traits[:2])}, which conflicts with {property_name.lower()}."
                )
            elimination_details.append(f"Objects other than {answer} do not demonstrate the physical traits tied to {property_name}.")

        if not elimination_details and negative_traits:
            elimination_details.append(
                f"{others_clause} {others_verb} show {', '.join(negative_traits[:2])}, which conflicts with {property_name.lower()}."
            )
        
        if elimination_details:
            steps.append(
                self._create_reasoning_step(
                    "elimination",
                    "Other choices are ruled out because " + " ".join(elimination_details)
                )
            )
        
        thinking_clause = self._compose_physical_property_thinking_sentence(
            answer,
            property_name,
            others_clause,
            others_verb,
            visual_clues,
            positive_traits,
        )
        steps.append(
            self._create_reasoning_step(
                "thinking",
                thinking_clause
            )
        )
        return steps

    def generate_repurposing_reasoning(
        self,
        question_type: str,
        target_object: str,
        object_set: List[str],
        answer: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate repurposing reasoning for any repurposing type with detailed elimination analysis."""
        # Try to use repurposing-specific template if available
        repurposing_key = question_type.replace('repurposing_', '').replace('_concept', '')
        template = self._find_template_match(repurposing_key, self.repurposing_templates)
        
        if template:
            objects_str = ', '.join(object_set)
            format_kwargs = {'objects': objects_str, 'answer': answer}
            import string
            template_placeholders = [field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name]
            filtered_kwargs = {k: v for k, v in format_kwargs.items() if k in template_placeholders}
            analysis_sentence = self.clean_text(template.format(**filtered_kwargs))
            qa_key = self._normalize_cluster_key(f"repurposing_{repurposing_key}_concept")
            qa_description = self.qa_space_descriptions.get(qa_key)
            feature_clause = None
            if qa_description:
                feature_clause = self._description_to_feature_clause(qa_description)
                if feature_clause and repurposing_key == "shield":
                    feature_clause = feature_clause.replace("width or length greater than height", "")
                    feature_clause = feature_clause.replace("  ", " ").replace(" ,", ", ").strip(" ,")
            if feature_clause:
                feature_clause = feature_clause.replace(",,", ",")
                feature_clause = feature_clause.replace(", ,", ", ")
                feature_clause = feature_clause.strip(" ,")
            steps: List[Dict[str, Any]] = [
                self._create_reasoning_step(
                    "question_analysis",
                    analysis_sentence,
                    input_data={"objects": object_set, "answer": answer},
                )
            ]
            if feature_clause:
                steps.append(
                    self._create_reasoning_step(
                        "question_analysis",
                        f"This concept expects {feature_clause}.",
                    )
                )
            steps.append(
                self._create_reasoning_step(
                    "thinking",
                    f"The {answer} offers the structure needed to accomplish the repurposed goal.",
                    input_data={"object": answer},
                    output_data={"answer": answer},
                )
            )
        else:
            # Use enhanced reasoning with physical properties (shape, rigidity, stability) for repurposing
            # Repurposing is about physical form and properties, not material composition
            steps = self._generate_taxonomy_enhanced_reasoning(
                question_type, target_object, object_set, answer, "physical"
            )
        return steps

    def generate_spatial_reasoning(
        self,
        question_type: str,
        target_object: str,
        object_set: List[str],
        answer: str,
        spatial_context: Optional[Dict] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate structured spatial reasoning for spatial question types with orientation emphasis."""
        self._last_spatial_answer_label = None

        reference_phrase = "the reference object"
        target_phrase = "the compared object"
        answer_label = answer
        relation_key_map = {
            'spatial_left_right': 'left_right',
            'spatial_front_behind': 'front_behind',
            'spatial_above_below': 'above_below',
            'spatial_closer_to_camera': 'closer'
        }
        relation_value = None
        calculation_details = None
            if spatial_context:
            reference_phrase = spatial_context.get('reference_phrase', reference_phrase)
            target_phrase = spatial_context.get('target_phrase', target_phrase)
            answer_label = spatial_context.get('answer_label', answer_label)
            if isinstance(spatial_context.get('relation'), str):
                relation_value = spatial_context['relation']
            spatial_rel = spatial_context.get('spatial_relationship')
            relation_key = relation_key_map.get(question_type)
            if not relation_value and relation_key and isinstance(spatial_rel, dict):
                relation_value = spatial_rel.get(relation_key)
            calculation_details = spatial_context.get("calculation_details")

        relation_aliases = {
            'spatial_left_right': {'left', 'right'},
            'spatial_front_behind': {'front', 'behind'},
            'spatial_above_below': {'above', 'below'},
            'spatial_closer_to_camera': {'closer', 'farther', 'target', 'reference'},
        }
        allowed_terms = relation_aliases.get(question_type, set())
        canonical_relation = (relation_value or "").lower()
        normalized_answer = (answer_label or "").lower()
        if question_type == 'spatial_closer_to_camera':
            if canonical_relation == 'target':
                canonical_relation = 'closer'
            elif canonical_relation == 'reference':
                canonical_relation = 'farther'
        if canonical_relation not in allowed_terms and normalized_answer in allowed_terms:
            canonical_relation = normalized_answer
        if normalized_answer not in allowed_terms and canonical_relation in allowed_terms:
            answer_label = canonical_relation
        relation_value = canonical_relation if canonical_relation else None

        axis_names = {
            'spatial_left_right': 'horizontal (x)',
            'spatial_front_behind': 'depth (z)',
            'spatial_above_below': 'vertical (y)',
            'spatial_closer_to_camera': 'depth (z)',
        }
        axis_name = axis_names.get(question_type, 'relevant')

        steps: List[Dict[str, Any]] = []

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

        def _compose_axis_description(delta_meta: Dict[str, Any], relation: Optional[str], default_text: Optional[str]) -> Optional[str]:
            if not delta_meta:
                return default_text
            relation_normalized = (relation or "").lower()
            dx = delta_meta.get("dx")
            dy = delta_meta.get("dy")

            def _format_offset(value: Optional[float], axis_word: str, relation_word: Optional[str], fallback_direction: Optional[str]) -> Optional[str]:
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

        if calculation_details:
            reference_info = calculation_details.get("reference", {}) or {}
            target_info = calculation_details.get("target", {}) or {}
            delta_info = calculation_details.get("delta", {}) or {}
            relation_value = calculation_details.get("relation_value", relation_value)
            if isinstance(relation_value, str):
                relation_value = relation_value.lower()
                if question_type == 'spatial_closer_to_camera':
                    if relation_value == 'target':
                        relation_value = 'closer'
                    elif relation_value == 'reference':
                        relation_value = 'farther'
            relation_clause = self._build_spatial_relation_clause(
                question_type, relation_value, target_phrase, reference_phrase, answer_label=answer_label
            )

            if reference_info:
                steps.append(
                    self._create_reasoning_step(
                        "orientation",
                        _format_orientation_line(reference_phrase, reference_info),
                        input_data=reference_info,
                    )
                )
            else:
                steps.append(
                    self._create_reasoning_step(
                        "orientation",
                        f"I anchor on {reference_phrase} to establish the scene orientation.",
                    )
                )

            if target_info:
                steps.append(
                    self._create_reasoning_step(
                        "orientation",
                        _format_orientation_line(target_phrase, target_info),
                        input_data=target_info,
                    )
                )
            else:
                steps.append(
                    self._create_reasoning_step(
                        "orientation",
                        f"I locate {target_phrase} relative to {reference_phrase} along the {axis_name} axis.",
                    )
                )

            axis_description = delta_info.get("axis_description")
            axis_description = _compose_axis_description(delta_info, relation_value, axis_description)
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
                self._create_reasoning_step(
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
                self._create_reasoning_step(
                    "thinking",
                    reasoning_summary,
                    output_data={"answer": answer_label, "relation": relation_value or "unknown"},
                )
            )
        else:
            relation_clause = self._build_spatial_relation_clause(
                question_type, relation_value, target_phrase, reference_phrase, answer_label=answer_label
            )
            steps.append(
                self._create_reasoning_step(
                    "orientation",
                    f"I anchor on {reference_phrase} to establish the scene orientation.",
                )
            )
            steps.append(
                self._create_reasoning_step(
                    "orientation",
                    f"I locate {target_phrase} relative to {reference_phrase} along the {axis_name} axis.",
                )
            )

            steps.append(
                self._create_reasoning_step(
                    "calculation",
                    relation_clause
                    and f"I compare bounding box centers along the {axis_name} axis: {relation_clause}"
                    or f"I compare bounding box centers along the {axis_name} axis to evaluate the relation between {target_phrase} and {reference_phrase}.",
                )
            )

            steps.append(
                self._create_reasoning_step(
                    "thinking",
                    f"These cues confirm that {answer_label} satisfies the spatial relationship while alternatives do not.",
                    output_data={"answer": answer_label},
                )
            )
        if relation_value in {"left", "right", "front", "behind", "above", "below", "closer", "farther"}:
            self._last_spatial_answer_label = relation_value
        else:
            self._last_spatial_answer_label = answer_label if answer_label else None

        return steps

    def generate_description_reasoning(
        self,
        question_type: str,
        target_object: str,
        object_set: List[str],
        answer: str,
        description: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate reasoning for description questions using a concise statement."""
        description_clean = self.clean_text(description) if description else "the description"
        if "non" in question_type.lower():
            sentence = f"The {answer} does not match the description {description_clean}."
        else:
            sentence = f"The {answer} matches the description {description_clean}."

        return [
            self._create_reasoning_step(
                "thinking",
                sentence,
                output_data={"answer": answer},
            )
        ]

    def generate_comprehensive_reasoning(
        self,
        question_type: str,
        target_object: str,
        object_set: List[str],
        answer: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive reasoning for any question type."""
        filter_chain = kwargs.pop('filter_chain', None)
        box_to_object = kwargs.pop('box_to_object', None)
        core_steps: Optional[List[Dict[str, Any]]] = None
        filter_steps: Optional[List[Dict[str, Any]]] = None

        if filter_chain and self.taxonomy_utils:
            filter_steps = self.generate_taxonomy_aware_reasoning(
                question_type, target_object, object_set, answer, filter_chain, **kwargs
            )
        if question_type.startswith('material_'):
                material_steps = self.generate_material_reasoning(
                    question_type, target_object, object_set, answer, **kwargs
                )
                filter_only = [
                    step for step in filter_steps if step.get("reasoning_type") == "filter_analysis"
                ]
                refined_material = [
                    step for step in material_steps if step.get("reasoning_type") != "filter_analysis"
                ]
                core_steps = filter_only + refined_material if filter_only else refined_material
            elif question_type == 'physical_property':
                physical_steps = self.generate_physical_property_reasoning(
                    question_type, target_object, object_set, answer, **kwargs
                )
                filter_only = [
                    step for step in filter_steps if step.get("reasoning_type") == "filter_analysis"
                ]
                refined_physical = [
                    step for step in physical_steps if step.get("reasoning_type") != "filter_analysis"
                ]
                core_steps = filter_only + refined_physical if filter_only else refined_physical
            else:
                core_steps = filter_steps

        if core_steps is None:
            if question_type.startswith('material_'):
                core_steps = self.generate_material_reasoning(
                    question_type, target_object, object_set, answer, **kwargs
                )
        elif question_type.startswith('affordance_'):
                core_steps = self.generate_affordance_reasoning(
                    question_type, target_object, object_set, answer, **kwargs
                )
            elif question_type.startswith('function_') or question_type.startswith('functional_'):
                core_steps = self.generate_function_reasoning(
                    question_type, target_object, object_set, answer, **kwargs
                )
            elif question_type == 'physical_property':
                core_steps = self.generate_physical_property_reasoning(
                    question_type, target_object, object_set, answer, **kwargs
                )
        elif question_type.startswith('repurposing_'):
                core_steps = self.generate_repurposing_reasoning(
                    question_type, target_object, object_set, answer, **kwargs
                )
        elif question_type.startswith('spatial_'):
                spatial_context = kwargs.pop('spatial_context', None)
                core_steps = self.generate_spatial_reasoning(
                    question_type, target_object, object_set, answer, spatial_context, **kwargs
                )
                if self._last_spatial_answer_label:
                    answer = self._last_spatial_answer_label
        elif question_type.startswith('description_'):
                description = kwargs.pop('description', 'the given description')
                core_steps = self.generate_description_reasoning(
                    question_type, target_object, object_set, answer, description, **kwargs
                )
            elif question_type.startswith('compositional_'):
                core_steps = self.generate_compositional_reasoning(
                    question_type, target_object, object_set, answer, **kwargs
                )
        else:
                core_steps = self._generate_taxonomy_enhanced_reasoning(
                    question_type, target_object, object_set, answer, "material"
                )

        if core_steps is None:
            core_steps = []

        return self.compose_structured_reasoning(
            question_type, object_set, answer, core_steps, box_to_object=box_to_object
        )

    def generate_taxonomy_aware_reasoning(
        self,
        question_type: str,
        target_object: str,
        object_set: List[str],
        answer: str,
        filter_chain: List[Dict],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate reasoning using taxonomy clustering information from filter chain."""
        analysis_lines: List[str] = []
        feature_highlights: List[str] = []

        for i, step in enumerate(filter_chain):
            filter_type = step.get("filter_type", "")
            filter_value = step.get("filter_value", "")
            objects_before = step.get("objects_before", [])
            objects_after = step.get("objects_after", [])
            eliminated_objects = [obj for obj in objects_before if obj not in objects_after]
            feature_summary = self._describe_cluster_features(filter_type, filter_value)
            if feature_summary:
                feature_highlights.append(feature_summary)

            if filter_type == "Unique":
                remaining = self._format_object_list(objects_after or objects_before)
                if remaining:
                    analysis_lines.append(f"Only {remaining} stays unique, pointing to {answer}.")
            else:
                trait_phrase = self._render_filter_trait(filter_type, filter_value, feature_summary)
                if eliminated_objects:
                    removed_list = self._format_object_list(eliminated_objects)
                    if removed_list:
                        analysis_lines.append(f"{removed_list.capitalize()} lack {trait_phrase}, so I set them aside.")
                elif objects_after:
                    kept_list = self._format_object_list(objects_after)
                    if kept_list:
                        analysis_lines.append(f"{kept_list.capitalize()} still show {trait_phrase}.")

        steps: List[Dict[str, Any]] = []
        if analysis_lines:
            steps.append(
                self._create_reasoning_step(
                    "analysis",
                    steps=[
                        {"label": f"step {idx+1}", "description": self.clean_text(sentence)}
                        for idx, sentence in enumerate(analysis_lines)
                    ],
                )
            )

        features_for_thinking = self._summarize_feature_list(feature_highlights)
        if features_for_thinking:
            thinking_clause = f"This leaves {answer} as the only object still showing {features_for_thinking}."
        else:
            thinking_clause = f"This leaves {answer} as the only object that meets the requirements."
        steps.append(
            self._create_reasoning_step(
                "thinking",
                thinking_clause
            )
        )
        return steps
    
    def _get_taxonomy_cluster_info(self, filter_type: str, filter_value: str, answer: str) -> str:
        """Get taxonomy cluster information for the filter"""
        if not self.taxonomy_utils:
            return ""
        
        try:
            # Get clusters for the answer object
            lower_type = filter_type.lower()
            clusters: Optional[List[str]] = None
            if lower_type == "material":
                clusters = self.taxonomy_utils.get_object_clusters(answer, 'final_taxonomy_material')
            elif lower_type == "affordance":
                clusters = self.taxonomy_utils.get_object_clusters(answer, 'final_taxonomy_affordances')
            elif lower_type == "function":
                clusters = self.taxonomy_utils.get_object_clusters(answer, 'final_taxonomy_function')
            elif lower_type == "physical_property":
                clusters = self.taxonomy_utils.get_object_clusters(answer, 'final_taxonomy_physical_properties')
                if clusters:
                    clusters = [
                        cluster for cluster in clusters
                        if cluster and not is_void_cluster(cluster, 'physical')
                    ]

            if clusters:
                cleaned_clusters = [self.clean_cluster_name(cluster) for cluster in clusters]
                feature_highlights = [
                    self._describe_cluster_features(filter_type, cluster) for cluster in clusters
                ]
                feature_summary = self._summarize_feature_list(
                    [item for item in feature_highlights if item]
                )
                cluster_desc = ""
                label = self._cluster_label(lower_type, plural=len(cleaned_clusters) > 1)
                if lower_type == "physical_property":
                    cluster_desc = f"has {label}: {', '.join(cleaned_clusters)}"
                else:
                    cluster_desc = f"belongs to {label}: {', '.join(cleaned_clusters)}"
                if feature_summary:
                    if cluster_desc:
                        cluster_desc = f"{cluster_desc} showing {feature_summary}"
                    else:
                        cluster_desc = f"shows {feature_summary}"
                return f" ({cluster_desc})"
        except Exception:
            pass
        
        return ""
    
    def _get_elimination_reason_with_taxonomy(
        self,
        filter_type: str,
        filter_value: str,
        eliminated_objects: List[str],
        feature_summary: Optional[str] = None,
    ) -> str:
        """Get detailed elimination reason using taxonomy information"""
        if not self.taxonomy_utils:
            objs_text = ', '.join(eliminated_objects)
            if feature_summary:
                return (
                    f"Objects {objs_text} were eliminated because they lack "
                    f"{feature_summary or 'the required characteristics'}."
                )
            return (
                f"Objects {objs_text} were eliminated because they don't match the filter criteria."
            )
        
        elimination_reasons = []
        lower_type = (filter_type or "").lower()
        label = self._cluster_label(lower_type)
        plural_label = self._cluster_label(lower_type, plural=True)
        
        for obj in eliminated_objects:
            try:
                # Get clusters for eliminated object
                if lower_type == "material":
                    clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_material')
                    if clusters:
                        cleaned_clusters = [self.clean_cluster_name(cluster) for cluster in clusters]
                        reason = f"{obj} ({label}: {', '.join(cleaned_clusters)})"
                    else:
                        reason = f"{obj} (no matching {label})"
                elif lower_type == "affordance":
                    clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_affordances')
                    if clusters:
                        cleaned_clusters = [self.clean_cluster_name(cluster) for cluster in clusters]
                        reason = f"{obj} ({label}: {', '.join(cleaned_clusters)})"
                    else:
                        reason = f"{obj} (no matching {label})"
                elif lower_type == "function":
                    clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_function')
                    if clusters:
                        cleaned_clusters = [self.clean_cluster_name(cluster) for cluster in clusters]
                        reason = f"{obj} ({label}: {', '.join(cleaned_clusters)})"
                    else:
                        reason = f"{obj} (no matching {label})"
                elif lower_type == "physical_property":
                    clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_physical_properties')
                    if clusters:
                        cleaned_clusters = [
                            self.clean_cluster_name(cluster)
                            for cluster in clusters
                            if cluster and not is_void_cluster(cluster, 'physical')
                        ]
                        if cleaned_clusters:
                            reason = f"{obj} ({plural_label}: {', '.join(cleaned_clusters)})"
                        else:
                            reason = f"{obj} (no relevant {plural_label})"
                    else:
                        reason = f"{obj} (no relevant {plural_label})"
                else:
                    reason = obj

                if feature_summary:
                    reason = f"{reason} lacks {feature_summary}"
                elimination_reasons.append(reason)
            except Exception:
                if feature_summary:
                    elimination_reasons.append(f"{obj} lacks {feature_summary}")
                else:
                    elimination_reasons.append(obj)
        
        if feature_summary:
            return (
                f"Eliminated objects: {', '.join(elimination_reasons)} because they lack "
                f"{feature_summary} required by the {filter_type} filter '{filter_value}'."
            )
        return (
            f"Eliminated objects: {', '.join(elimination_reasons)} because they don't match the "
            f"{filter_type} criteria '{filter_value}'."
        )

    def _generate_taxonomy_enhanced_reasoning(
        self,
        question_type: str,
        target_object: str,
        object_set: List[str],
        answer: str,
        category: str,
    ) -> List[Dict[str, Any]]:
        """Generate taxonomy-enhanced reasoning using clustering information with detailed elimination analysis."""
        step_type = f"{category}_analysis" if category else "analysis"
            objects_str = ', '.join(object_set)

        if not self.taxonomy_utils:
            return [
                self._create_reasoning_step(
                    step_type,
                    f"I compare the taxonomy-driven attributes of {objects_str} to match the question criteria.",
                    input_data={"objects": object_set},
                ),
                self._create_reasoning_step(
                    "thinking",
                    f"The {answer} best satisfies those attributes while the alternatives do not.",
                    output_data={"answer": answer},
                ),
            ]

        try:
            feature_highlights: List[str] = []
            clusters: List[str] = []
            feature_filter_lookup = {
                "affordance": "affordance",
                "material": "material",
                "function": "function",
                "physical": "physical_property",
            }

            if category == "affordance":
                clusters = self.taxonomy_utils.get_object_clusters(answer, 'final_taxonomy_affordances') or []
            elif category == "material":
                clusters = self.taxonomy_utils.get_object_clusters(answer, 'final_taxonomy_material') or []
            elif category == "function":
                clusters = self.taxonomy_utils.get_object_clusters(answer, 'final_taxonomy_function') or []
            elif category == "physical":
                raw_clusters = self.taxonomy_utils.get_object_clusters(answer, 'final_taxonomy_physical_properties') or []
                clusters = [
                    cluster for cluster in raw_clusters
                    if cluster and not is_void_cluster(cluster, 'physical')
                ]

            selected_cluster = clusters[0] if clusters else None

            if clusters:
                filter_key = feature_filter_lookup.get(category)
                for cluster in clusters:
                    feature_phrase = self._describe_cluster_features(filter_key or category, cluster)
                    if feature_phrase:
                        feature_highlights.append(feature_phrase)

            elimination_steps: List[str] = []
            required_features = self._summarize_feature_list(feature_highlights)
            for obj in object_set:
                if obj == answer:
                    continue
                elimination_steps.append(
                    self._analyze_object_for_elimination(obj, category, question_type, required_features)
                )

            steps: List[Dict[str, Any]] = []

            category_label = category.replace('_', ' ') if category else "taxonomy"
            cluster_label = self.clean_cluster_name(selected_cluster) if selected_cluster else None
            naturalized_cluster = self._naturalize_cluster_phrase(selected_cluster, category) if selected_cluster else None
            qa_desc = self.qa_space_descriptions.get(self._normalize_cluster_key(question_type), "")
            qa_desc_clean = self.clean_text(qa_desc) if qa_desc else ""
            if qa_desc_clean and cluster_label and naturalized_cluster:
                pattern = re.compile(re.escape(cluster_label), flags=re.IGNORECASE)
                qa_desc_clean = pattern.sub(naturalized_cluster, qa_desc_clean)

            feature_summary = required_features or self._summarize_feature_list(feature_highlights)
            feature_summary = self.clean_text(feature_summary) if feature_summary else ""

            others = [obj for obj in object_set if obj != answer]
            others_text = ", ".join(others)

            analysis_sentences: List[str] = []
            if naturalized_cluster and feature_summary:
                analysis_sentences.append(
                    f"The {category_label} focus is {naturalized_cluster}, highlighting {feature_summary}."
                )
            elif naturalized_cluster:
                analysis_sentences.append(
                    f"The {category_label} focus is {naturalized_cluster}."
                )
            elif feature_summary:
                analysis_sentences.append(f"This question focuses on objects with {feature_summary}.")

            if qa_desc_clean:
                analysis_sentences.append(qa_desc_clean)

            analysis_sentences.append(f"The {answer} demonstrates these characteristics.")
            if others_text:
                analysis_sentences.append(f"Other candidates ({others_text}) do not.")

            if not analysis_sentences:
                analysis_sentences.append(f"The {answer} satisfies the {category_label} requirement while others do not.")

            steps.append(
                self._create_reasoning_step(
                    step_type,
                    steps=[
                        {"label": f"step {idx+1}", "description": sentence}
                        for idx, sentence in enumerate(analysis_sentences)
                    ],
                    input_data={
                        "objects": object_set,
                        "answer": answer,
                    },
                )
            )

            if elimination_steps:
                steps.append(
                    self._create_reasoning_step(
                        "elimination",
                        "Other choices are ruled out because " + ' '.join(elimination_steps),
                        input_data={"eliminations": elimination_steps},
                    )
                )

            thinking_sentence = f"Therefore the {answer} best satisfies the question's criteria"
            if required_features:
                thinking_sentence = (
                    f"Therefore the {answer} best satisfies the question's criteria with {required_features}"
                )
            if question_type == "counterfactual_water":
                if required_features:
                    thinking_sentence += f" Because it retains {required_features}, it is the most susceptible to water damage."
                else:
                    thinking_sentence += " This material is most susceptible to water damage."
            elif question_type == "counterfactual_heat":
                if required_features:
                    thinking_sentence += f" Because it retains {required_features}, it is the most susceptible to heat damage."
                else:
                    thinking_sentence += " This material is most susceptible to heat damage."
            steps.append(
                self._create_reasoning_step(
                    "thinking",
                    thinking_sentence,
                    output_data={"answer": answer, "features": required_features},
                )
            )
            return steps

        except Exception:
            return [
                self._create_reasoning_step(
                    step_type,
                    f"I review {objects_str} against the required taxonomy features.",
                    input_data={"objects": object_set},
                ),
                self._create_reasoning_step(
                    "thinking",
                    f"The {answer} remains the only option that fits those features.",
                    output_data={"answer": answer},
                ),
            ]
    
    def _analyze_object_for_elimination(
        self,
        obj: str,
        category: str,
        question_type: str,
        required_features: Optional[str] = None,
    ) -> str:
        """Analyze why an object is eliminated and what attributes it lacks"""
        try:
            # Get clusters for the object
            obj_profiles: List[str] = []
            label = self._cluster_label(category)
            plural_label = self._cluster_label(category, plural=True)
            if category == "affordance":
                clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_affordances')
                if clusters:
                    natural = [
                        self._naturalize_cluster_phrase(cluster, category) or self.clean_cluster_name(cluster)
                        for cluster in clusters
                    ]
                    natural = [phrase for phrase in natural if phrase]
                    if natural:
                        obj_profiles.append(f"{plural_label}: {', '.join(natural)}")
            elif category == "material":
                clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_material')
                if clusters:
                    natural = [
                        self._naturalize_cluster_phrase(cluster, category) or self.clean_cluster_name(cluster)
                        for cluster in clusters
                    ]
                    natural = [phrase for phrase in natural if phrase]
                    if natural:
                        obj_profiles.append(f"{plural_label}: {', '.join(natural)}")
            elif category == "function":
                clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_function')
                if clusters:
                    natural = [
                        self._naturalize_cluster_phrase(cluster, category) or self.clean_cluster_name(cluster)
                        for cluster in clusters
                    ]
                    natural = [phrase for phrase in natural if phrase]
                    if natural:
                        obj_profiles.append(f"{plural_label}: {', '.join(natural)}")
            elif category == "physical":
                clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_physical_properties')
                if clusters:
                    filtered_clusters = [
                        cluster for cluster in clusters
                        if cluster and not is_void_cluster(cluster, 'physical')
                    ]
                    natural_names: List[str] = []
                    feature_bits: List[str] = []
                    for cluster in filtered_clusters:
                        natural_name = self._naturalize_cluster_phrase(cluster, category) or self.clean_cluster_name(cluster)
                        if natural_name:
                            natural_names.append(natural_name)
                        described = self._describe_cluster_features('physical_property', cluster)
                        if described:
                            feature_bits.append(described)
                    feature_summary = self._summarize_feature_list(feature_bits)
                    description_parts: List[str] = []
                    if natural_names:
                        description_parts.append(f"{plural_label}: {', '.join(natural_names)}")
                    if feature_summary:
                        description_parts.append(feature_summary)
                    if description_parts:
                        obj_profiles.append(" ".join(description_parts))
            
            # Determine what attributes the object lacks for this question type
            missing_attributes = self._get_missing_attributes_for_question(obj, question_type, category)
            
            if obj_profiles and missing_attributes:
                reason = f"{obj} ({'; '.join(obj_profiles)}) lacks {missing_attributes}"
            elif obj_profiles:
                reason = f"{obj} ({'; '.join(obj_profiles)}) does not have the required characteristics"
            elif missing_attributes:
                reason = f"{obj} lacks {missing_attributes}"
            else:
                reason = f"{obj} has no recorded {plural_label} that satisfy the criteria"

            if required_features:
                if reason.endswith('.'):
                    reason = reason.rstrip('.')
                reason += f" and does not offer {required_features}"
            return reason
                
        except Exception:
            base = f"{obj} does not meet the criteria"
            if required_features:
                base += f" for {required_features}"
            return base
    
    def _get_missing_attributes_for_question(self, obj: str, question_type: str, category: str) -> str:
        """Determine what specific attributes an object lacks for a given question type"""
        # Map question types to required attributes
        attribute_requirements = {
            # Repurposing questions
            'repurposing_stepstool_concept': {
                'required': 'stable structure, appropriate height, load-bearing capacity',
                'material_focus': 'rigid materials like wood, metal, or plastic'
            },
            'repurposing_cushion_concept': {
                'required': 'soft materials, compressible properties, comfort',
                'material_focus': 'textiles, foam, or soft materials'
            },
            'repurposing_container_concept': {
                'required': 'hollow interior, opening or lid, containment space',
                'material_focus': 'any material that can form containers'
            },
            'repurposing_reflector_concept': {
                'required': 'smooth surface, reflective properties, flat or curved surface',
                'material_focus': 'metals, glass, or polished materials'
            },
            
            'repurposing_bookend_concept': {
                'required': 'weight, appropriate shape, stability',
                'material_focus': 'heavy materials like metal, stone, or dense materials'
            },
            
            'repurposing_tool_concept': {
                'required': 'appropriate shape, material properties, functional design',
                'material_focus': 'materials suitable for tool use'
            },
            
            # Affordance questions
            'affordance_furniture': {
                'required': 'furniture-related affordances like seating, storage, or support',
                'material_focus': 'materials suitable for furniture construction'
            },
            'affordance_wearables_and_apparel': {
                'required': 'body-fitting shape, fasteners, flexible material',
                'material_focus': 'textiles, flexible materials'
            },
            'affordance_contain_carry_package': {
                'required': 'hollow interior, carrying capability, containment space',
                'material_focus': 'materials that can form containers'
            },
            'affordance_sit_ride_attend': {
                'required': 'flat horizontal surface, appropriate height, stability',
                'material_focus': 'rigid materials for structural support'
            },
            
            # Function questions
            'function_seating': {
                'required': 'seating functionality, appropriate height, stability',
                'material_focus': 'materials suitable for seating'
            },
            'function_storage': {
                'required': 'storage capacity, organization capability, accessibility',
                'material_focus': 'materials suitable for storage containers'
            },
            'function_display': {
                'required': 'display surface, visibility, presentation capability',
                'material_focus': 'materials suitable for display surfaces'
            }
        }
        
        # Get the specific requirements for this question type
        requirements = attribute_requirements.get(question_type, {})
        if requirements:
            required_attrs = requirements.get('required', 'required characteristics')
            material_focus = requirements.get('material_focus', 'appropriate materials')
            return f"{required_attrs} and {material_focus}"
        
        # Generic fallback based on category
        if category == "material":
            return "the required material properties"
        elif category == "affordance":
            return "the required affordance capabilities"
        elif category == "function":
            return "the required functional properties"
        else:
            return "the required characteristics"

    def generate_compositional_reasoning(
        self,
        question_type: str,
        target_object: str,
        object_set: List[str],
        answer: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate structured reasoning for compositional questions."""
        steps: List[Dict[str, Any]] = []
        requirement_map = {
            "compositional_set_subtraction_container": {
                "description": "being rigid and movable while not functioning as a container",
                "required": ["rigid", "movable"],
                "forbidden": ["contain"],
            },
            "compositional_set_subtraction_hollow": {
                "description": "having a hollow structure",
                "required": ["hollow"],
                "forbidden": [],
            },
        }

        def format_list(items: List[str]) -> str:
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            if len(items) == 2:
                return f"{items[0]} and {items[1]}"
            return ", ".join(items[:-1]) + f", and {items[-1]}"

        if question_type == 'compositional_counterfactual_material':
            target = kwargs.get('target', target_object)
            to_material = kwargs.get('to_material', 'a different material')
            from_material = kwargs.get('from_material', 'its current material')
            steps.append(
                self._create_reasoning_step(
                    "counterfactual_analysis",
                    f"I compare {target}'s current material {from_material} against the proposed {to_material} to project changes in weight, durability, and functionality.",
                    input_data={"target": target, "from_material": from_material, "to_material": to_material},
                )
            )
            steps.append(
                self._create_reasoning_step(
                    "thinking",
                    f"These changes explain why the scenario affects the outcome involving {answer}.",
                    output_data={"answer": answer},
                )
            )
            return steps

        elif question_type == 'compositional_action_based':
            target = kwargs.get('target', target_object)
            goal = kwargs.get('goal', 'the specified task')
            steps.append(
                self._create_reasoning_step(
                    "action_analysis",
                    f"I evaluate {target}'s geometry and material composition to see if it can accomplish '{goal}'.",
                    input_data={"target": target, "goal": goal},
                )
            )
            steps.append(
                self._create_reasoning_step(
                    "thinking",
                    f"The {answer} provides the necessary affordances for '{goal}' while alternatives lack them.",
                    output_data={"answer": answer},
                )
            )
            return steps

        elif question_type == 'compositional_set_subtraction_dynamic':
            description = kwargs.get('description', 'the specified properties')
            options = kwargs.get('options', ', '.join(object_set))
            steps.append(
                self._create_reasoning_step(
                    "property_analysis",
                    f"I filter objects from {options} that satisfy '{description}'.",
                    input_data={"description": description, "options": options},
                )
            )
            steps.append(
                self._create_reasoning_step(
                    "thinking",
                    f"The {answer} uniquely satisfies the criteria while other objects either miss properties or conflict with them.",
                    output_data={"answer": answer},
                )
            )
            return steps

        else:
            requirement = requirement_map.get(question_type)
            if requirement:
                desc = requirement["description"]
                steps.append(
                    self._create_reasoning_step(
                        "composition_analysis",
                        f"I evaluate each object for {desc}.",
                        input_data={"requirement": desc},
                    )
                )

                sentences: List[str] = []
                required_props = requirement.get("required", [])
                forbidden_props = requirement.get("forbidden", [])

                required_feature_clauses = [
                    self._describe_requirement_trait(prop) for prop in required_props
                ]
                required_feature_summary = self._summarize_feature_list(
                    [clause for clause in required_feature_clauses if clause]
                )
                if required_feature_summary:
                    sentences.append(
                        f"The goal requires {format_list(required_props)} qualities, signaled by {required_feature_summary}."
                    )

                satisfied_props = [
                    prop for prop in required_props
                    if self.compositional_utils.taxonomy_utils.has_property(answer, prop)
                ]
                if satisfied_props:
                    satisfied_features = self._summarize_feature_list(
                        [
                            self._describe_requirement_trait(prop)
                            for prop in satisfied_props
                            if self._describe_requirement_trait(prop)
                        ]
                    )
                    if satisfied_features:
                        sentences.append(
                            f"The {answer} satisfies {format_list(satisfied_props)} by showing {satisfied_features}."
                        )
                    else:
                        sentences.append(
                            f"The {answer} exhibits {format_list(satisfied_props)} qualities."
                        )
                if forbidden_props:
                    violates = [
                        prop for prop in forbidden_props
                        if self.compositional_utils.taxonomy_utils.has_property(answer, prop)
                    ]
                    if not violates:
                        forbidden_features = self._summarize_feature_list(
                            [
                                self._describe_requirement_trait(prop)
                                for prop in forbidden_props
                                if self._describe_requirement_trait(prop)
                            ]
                        )
                        if forbidden_features:
                            sentences.append(
                                f"The {answer} also avoids {format_list(forbidden_props)} cues such as {forbidden_features}."
                            )
                        else:
                            sentences.append(f"The {answer} avoids {format_list(forbidden_props)} usage.")

                failing_descriptions: List[str] = []
                for obj in object_set:
                    if obj == answer:
                        continue
                    display_name = obj if obj.lower().startswith(("the ", "a ", "an ")) else f"the {obj}"
                    missing = [
                        prop for prop in required_props
                        if not self.compositional_utils.taxonomy_utils.has_property(obj, prop)
                    ]
                    violations = [
                        prop for prop in forbidden_props
                        if self.compositional_utils.taxonomy_utils.has_property(obj, prop)
                    ]
                    missing_features = self._summarize_feature_list(
                        [
                            self._describe_requirement_trait(prop)
                            for prop in missing
                            if self._describe_requirement_trait(prop)
                        ]
                    )
                    violation_features = self._summarize_feature_list(
                        [
                            self._describe_requirement_trait(prop)
                            for prop in violations
                            if self._describe_requirement_trait(prop)
                        ]
                    )
                    if missing and violations:
                        description = (
                            f"{display_name} lacks {format_list(missing)}"
                            + (f" (missing {missing_features})" if missing_features else "")
                            + f" and shows {format_list(violations)} traits"
                            + (f" ({violation_features})" if violation_features else "")
                        )
                        failing_descriptions.append(description)
                    elif missing:
                        description = f"{display_name} lacks {format_list(missing)}"
                        if missing_features:
                            description += f" (missing {missing_features})"
                        failing_descriptions.append(description)
                    elif violations:
                        description = f"{display_name} shows {format_list(violations)} traits"
                        if violation_features:
                            description += f" ({violation_features})"
                        failing_descriptions.append(description)

                if failing_descriptions:
                    sentences.append("Other options fail the test: " + ", ".join(failing_descriptions) + ".")

                if not sentences:
                    sentences.append(
                        f"The {answer} best fits the compositional requirements while the others do not."
                    )

                steps.append(
                    self._create_reasoning_step(
                        "thinking",
                        " ".join(sentences),
                        input_data={"required": required_props, "forbidden": forbidden_props},
                        output_data={"answer": answer},
                    )
                )
                return steps

            objects_str = ', '.join(object_set)
            steps.append(
                self._create_reasoning_step(
                    "composition_analysis",
                    f"I analyze relationships among {objects_str} to see which combination matches the criteria.",
                    input_data={"objects": object_set},
                )
            )
            steps.append(
                self._create_reasoning_step(
                    "thinking",
                    f"The {answer} best fits the compositional requirements while the others do not.",
                    output_data={"answer": answer},
                )
            )
            return steps

    def get_available_templates(self) -> Dict[str, List[str]]:
        """Get list of available templates for each category"""
        return {
            'materials': list(self.material_templates.keys()),
            'affordances': list(self.affordance_templates.keys()),
            'properties': list(self.property_templates.keys()),
            'functions': list(self.function_templates.keys()),
            'repurposing': list(self.repurposing_templates.keys()),
            'spatial': list(self.spatial_templates.keys())
        }

    def add_custom_template(self, category: str, key: str, template_data: Dict[str, Any]) -> None:
        """Add a custom template to the generator"""
        if category == 'material':
            self.material_templates[key] = template_data
        elif category == 'affordance':
            self.affordance_templates[key] = template_data
        elif category == 'property':
            self.property_templates[key] = template_data
        elif category == 'function':
            self.function_templates[key] = template_data
        elif category == 'repurposing':
            self.repurposing_templates[key] = template_data
        elif category == 'spatial':
            self.spatial_templates[key] = template_data
        else:
            raise ValueError(f"Unknown category: {category}")
