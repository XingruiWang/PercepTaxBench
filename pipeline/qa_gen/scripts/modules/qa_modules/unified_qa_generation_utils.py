import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

from modules.qa_modules.question_templates import get_question_template
from modules.qa_modules.cot_reasoning_utils import CoTReasoningGenerator
from modules.qa_modules.filter_utils import is_void_cluster, HIGH_RELIABILITY_CLASSES
from modules.qa_modules.compositional_utils import CompositionalUtils

logger = logging.getLogger(__name__)


class UnifiedQAGenerationUtils:
    """Unified QA generation logic for both real and sim images"""
    
    # Question type categorization constants
    HARD_QUESTION_PREFIXES = ['repurposing_', 'counterfactual_', 'compositional_', 'latent_']
    EASY_QUESTION_TYPES = [
        'description_matching',
        'function_knowledge',
        'material_property',
        'physical_property',
        'material_sound_absorption',
        'material_thermal_touch',
        'material_scratch_resistance',
        'affordance_',
        'functional_',
    ]  # Note: functional_* maps to "capability" category
    SPATIAL_QUESTION_TYPES = ['spatial_above_below', 'spatial_left_right', 'spatial_front_behind', 'spatial_closer_to_camera']
    
    # Taxonomy cluster mappings for question types
    DISABLED_QUESTION_TYPES = {'affordance_sit__ride__attend'}

    FUNCTION_EASY_TARGET = 2
    MATERIAL_EASY_TARGET = 1
    PHYSICAL_EASY_TARGET = 1
    DESCRIPTION_EASY_LIMIT = 1
    MATERIAL_INFERENCE_TYPES = {
        'material_sound_absorption',
        'material_thermal_touch',
        'material_scratch_resistance',
    }

    AFFORDANCE_CLUSTER_MAP = {
        'affordance_furniture': 'Furniture',
        'affordance_contain__carry__package': 'Contain / Carry / Package',
        'affordance_grip__carry__operate': 'Grip / Carry / Operate',
        'affordance_operate__use_device': 'Operate / Use Device',
        'affordance_mechanical_control': 'Mechanical Control',
        'affordance_mediated_action_and_meaning': 'Mediated Action & Meaning',
        'affordance_food_—_ingredients_and_produce': 'Food — Ingredients & Produce',
        'affordance_food_—_prepared_dishes': 'Food — Prepared Dishes',
        'affordance_cleaning_and_sanitation': 'Cleaning and Sanitation',
        'affordance_control__express__light': 'Control / Express / Light',
        'affordance_grow__plant_(vegetation)': 'Grow / Plant (Vegetation)',
        'affordance_architectural_components_and_fixtures': 'Architectural Components & Fixtures',
        'affordance_art_display_(view_appraise)': 'Art Display (View/Appraise)',
        'affordance_build__span__occupy': 'Build / Span / Occupy',
        'affordance_display__exhibit__signal_value': 'Display / Exhibit / Signal Value',
        'affordance_enclosures_and_venues_(enter_use)': 'Enclosures & Venues (Enter/Use)',
        'affordance_household__facility_operations': 'Household / Facility Operations',
        'affordance_interact_with_living_moving_things': 'Interact with Living/Moving Things',
        'affordance_place__support__work_on': 'Place / Support / Work On',
        'affordance_structured_operational_engagement': 'Structured Operational Engagement',
        'affordance_tableware_and_serveware': 'Tableware & Serveware',
        'affordance_wearables_and_apparel': 'Wearables & Apparel',
    }
    
    MATERIAL_CLUSTER_MAP = {
        'material_metals_and_alloys': 'Metals & Alloys',
        'material_textiles_fibers_and_leather': 'Textiles, Fibers & Leather',
        'material_wood_and_plant_based_solids': 'Wood & Plant‑Based Solids',
        'material_plastics_rubber_and_polymers': 'Plastics, Rubber & Polymers',
        'material_paper_cardboard_and_pulp': 'Paper, Cardboard & Pulp',
        'material_glass_and_transparent_(silicate)': 'Glass & Transparent (Silicate)',
        'material_biological_(plants/flowers)': 'Biological (Plants/Flowers)',
        'material_organic_food_and_edible_matter': 'Organic Food & Edible Matter',
        'material_biological_(animals/body_parts)': 'Biological (Animals/Body Parts)',
        'material_ceramics_porcelain_and_earthenware': 'Ceramics, Porcelain & Earthenware',
        'material_composites_and_multi_material_products': 'Composites & Multi‑Material Products',
        'material_stone_concrete_and_mineral': 'Stone, Concrete & Mineral',
        'material_liquids_and_semi_liquids': 'Liquids & Semi‑liquids',
        'material_gases_vapors_and_atmospheric': 'Gases, Vapors & Atmospheric',
    }
    
    def __init__(self, taxonomy_utils, object_utils, question_generation_utils, annotations_dir: Path = None, data_loading_utils=None):
        self.taxonomy_utils = taxonomy_utils
        self.object_utils = object_utils
        self.question_generation_utils = question_generation_utils
        self.annotations_dir = annotations_dir
        self.data_loading_utils = data_loading_utils
        self.compositional_utils = CompositionalUtils(taxonomy_utils)
    
    def generate_qa_from_space(self, image_id: str, available_objects: List[str], qa_space_data: Dict[str, Any], use_sm_valid_only: bool = False, max_questions_per_image: int = 8) -> List[Dict[str, Any]]:
        """
        Generate QA questions for an image using QA space analysis.
        
        IMPORTANT: The order of available_objects MUST match the detection order used for bbox images.
        This ensures colored box assignments (Red, Green, Blue...) match the bbox colors in the image.
        Colors are assigned by index: 0=red, 1=green, 2=blue, 3=yellow, 4=magenta, 5=cyan, etc.
        
        Args:
            image_id: Unique identifier for the image
            available_objects: List of object names available in the image (ordered by detection)
            qa_space_data: QA space analysis data loaded from JSON
            use_sm_valid_only: If True, only use sm_valid_objects (for sim images). If False, prioritize openimages_valid_objects (for real images).
            max_questions_per_image: Maximum number of questions to generate per image (default: 8)
        
        Returns:
            List of question dictionaries with answer, reasoning, and metadata
        """
        questions = []
        total_questions = 0
        easy_question_cap = 4
        
        question_answer_mappings = qa_space_data.get('question_answer_mappings', {})
        
        # Strategy 4: Prioritize hard questions - separate hard and easy question types
        hard_question_types = {k: v for k, v in question_answer_mappings.items() 
                             if any(k.startswith(prefix) for prefix in self.HARD_QUESTION_PREFIXES)}
        easy_question_types = {k: v for k, v in question_answer_mappings.items() 
                             if not any(k.startswith(prefix) for prefix in self.HARD_QUESTION_PREFIXES)}
        
        # Step 1: Loop through each object and find questions where it's in the answer space
        logger.info(f"Processing {len(available_objects)} objects: {available_objects}")
        logger.info(f"QA space has {len(question_answer_mappings)} question types ({len(hard_question_types)} hard, {len(easy_question_types)} easy)")
        
        # Collect all questions per object for filtering (Strategy 1)
        all_object_questions = {}  # target_object -> list of questions
        seen_question_texts = set()  # Track generated questions to prevent duplicates
        
        for target_object in available_objects:
            # Skip objects that are in void clusters for question-type-specific checks
            # (These checks are already done per question type, but we can skip early for efficiency)
            logger.info(f"Processing object: {target_object}")
            for question_key, question_data in question_answer_mappings.items():
                question_type = question_key
                if question_type in self.DISABLED_QUESTION_TYPES:
                    continue
                
                # Strategy 2: Skip unsupported material variants (keep material_property + inference types)
                if (
                    question_type.startswith('material_')
                    and question_type != 'material_property'
                    and question_type not in self.MATERIAL_INFERENCE_TYPES
                ):
                    continue  # Skip variants like material_textiles_*, material_metals_*, etc.
                
                # Use appropriate valid objects list based on data source
                if use_sm_valid_only:
                    # For sim images: only use sm_valid_objects
                    valid_objects = question_data.get('sm_valid_objects', [])
                else:
                    # For real images: prioritize openimages_valid_objects
                    valid_objects = question_data.get('openimages_valid_objects', question_data.get('sm_valid_objects', []))
                
                # Check if this object is in the answer space for this question type
                if target_object in valid_objects:
                    # Check if other objects in the scene are in the same answer space (conflict detection)
                    other_objects_in_scene = self._get_other_objects(available_objects, target_object)
                    conflicting_objects = [obj for obj in other_objects_in_scene if obj in valid_objects]
                    
                    # Generate question only if there are NO conflicting answers for this question type
                    should_generate = len(conflicting_objects) == 0
                    
                    # Additional conflict detection for compositional questions
                    # Check if ALL objects in scene have the required property (even if excluded from valid_objects)
                    # This prevents questions like "Which object is hollow?" when all objects are hollow
                    if should_generate and question_type.startswith('compositional_'):
                        if question_type == 'compositional_set_subtraction_hollow':
                            # Check if all objects in scene are hollow (regardless of container status)
                            all_hollow = all(
                                self.compositional_utils.check_properties(
                                    obj, ['hollow'], []
                                ) for obj in available_objects
                            )
                            if all_hollow:
                                logger.debug(f"Skipping {question_type} for {target_object}: all objects in scene are hollow")
                                should_generate = False
                        elif question_type == 'compositional_set_subtraction_container':
                            # Check if all objects in scene are rigid and movable (regardless of container/hollow status)
                            all_rigid_movable = all(
                                self.compositional_utils.check_properties(
                                    obj, ['rigid', 'movable'], []
                                ) for obj in available_objects
                            )
                            if all_rigid_movable:
                                logger.debug(f"Skipping {question_type} for {target_object}: all objects in scene are rigid and movable")
                                should_generate = False
                    
                    # Additional filtering for description_matching questions (void clusters + conflict detection)
                    if should_generate and question_type == 'description_matching':
                        # Step 1: Filter by void clusters
                        if self._is_object_in_any_void_cluster(target_object):
                            logger.debug(f"Skipping description_matching for {target_object}: object is in void cluster")
                            should_generate = False
                        
                        # Step 2: Check for description conflicts
                        if should_generate:
                            other_objects_in_scene = self._get_other_objects(available_objects, target_object)
                            if self._has_description_conflict(target_object, other_objects_in_scene):
                                logger.debug(f"Skipping description_matching for {target_object}: description conflicts with other objects")
                                should_generate = False
                    
                    # Additional filtering for material_property questions (void clusters + material uniqueness + taxonomy cluster conflicts)
                    if should_generate and question_type == 'material_property':
                        if self._is_object_in_void_cluster(target_object, 'material'):
                            logger.debug(f"Skipping material_property for {target_object}: object is in void cluster")
                            should_generate = False
                        else:
                            # Check material uniqueness to avoid ambiguity (checks material string)
                            if not self._is_material_unique(target_object, available_objects):
                                should_generate = False
                            # Also check taxonomy cluster conflicts (checks material cluster)
                            elif self._has_taxonomy_cluster_conflict(target_object, available_objects, question_type):
                                should_generate = False
                    
                    # Taxonomy cluster conflict checking for taxonomy-specific questions
                    if should_generate and self._is_taxonomy_based_question(question_type):
                        if self._has_taxonomy_cluster_conflict(target_object, available_objects, question_type):
                            should_generate = False
                    
                    if should_generate:
                        # Initialize spatial variables for all questions
                        spatial_relationship = None
                        reference_object = None
                        
                        # Create question data dict for this question
                        question_data_dict = question_data.copy()
                        
                        type_questions = self.question_generation_utils.generate_questions_for_type(
                            question_type, [target_object], available_objects, question_data_dict, image_id,
                            spatial_relationship, reference_object
                        )
                        if type_questions:
                            # Filter out duplicates based on question text before storing
                            unique_type_questions = []
                            for q in type_questions:
                                question_text = q.get('question', '').strip()
                                if question_text and question_text not in seen_question_texts:
                                    seen_question_texts.add(question_text)
                                    unique_type_questions.append(q)
                                else:
                                    logger.debug(f"Skipping duplicate question for {image_id}: {question_text[:50]}...")
                            
                            if unique_type_questions:
                                # Store questions per object for later filtering (Strategy 1)
                                if target_object not in all_object_questions:
                                    all_object_questions[target_object] = []
                                all_object_questions[target_object].extend(unique_type_questions)
        
        # Strategy 1: Limit easy questions to 2 total per image (keep all hard questions)
        # Separate spatial questions for special handling (Strategy 3)
        # Collect spatial questions separately for limiting
        spatial_questions_from_qa_space = []
        all_easy_questions_from_qa_space = []
        non_spatial_questions_from_qa_space = []
        
        for target_object, object_questions_list in all_object_questions.items():
            # Separate questions by type
            spatial_qs = self._filter_questions_by_type(object_questions_list, self.SPATIAL_QUESTION_TYPES)
            easy_questions = self._filter_questions_by_type(object_questions_list, self.EASY_QUESTION_TYPES)
            hard_questions = self._filter_questions_by_type(object_questions_list, self.HARD_QUESTION_PREFIXES, use_prefix=True)
            
            # Keep ALL hard questions
            non_spatial_questions_from_qa_space.extend(hard_questions)
            
            # Collect spatial questions separately
            spatial_questions_from_qa_space.extend(spatial_qs)
            
            # Collect all easy questions from QA space (will be limited later)
            all_easy_questions_from_qa_space.extend(easy_questions)
            logger.debug(f"Object {target_object}: Collected {len(easy_questions)} easy questions, kept {len(hard_questions)} hard questions")
        
        # Limit easy questions from QA space with priority for function/material coverage
        if all_easy_questions_from_qa_space:
            max_easy = min(easy_question_cap, len(all_easy_questions_from_qa_space))
            selected_easy = self._select_easy_questions(all_easy_questions_from_qa_space, max_easy)
            non_spatial_questions_from_qa_space.extend(selected_easy)
            logger.debug(
                "Selected %d easy questions (function=%d, material=%d, physical=%d, material_inference=%d, functional_seating=%d, description=%d, other=%d) from %d total (QA space)",
                len(selected_easy),
                sum(1 for q in selected_easy if q.get('question_type') == 'function_knowledge'),
                sum(1 for q in selected_easy if q.get('question_type') == 'material_property'),
                sum(1 for q in selected_easy if q.get('question_type') == 'physical_property'),
                sum(1 for q in selected_easy if q.get('question_type') in self.MATERIAL_INFERENCE_TYPES),
                sum(1 for q in selected_easy if q.get('question_type') == 'functional_seating'),
                sum(1 for q in selected_easy if q.get('question_type') == 'description_matching'),
                sum(1 for q in selected_easy if q.get('question_type') not in {'function_knowledge', 'material_property', 'physical_property', 'functional_seating', 'description_matching'} | self.MATERIAL_INFERENCE_TYPES),
                len(all_easy_questions_from_qa_space)
            )
        
        # Strategy 3: Generate additional questions for objects not in QA space
        logger.info(f"Generating additional questions for objects not in QA space: {available_objects}")
        additional_questions = self.generate_additional_questions_not_in_qa_space(available_objects, image_id)
        logger.info(f"Generated {len(additional_questions)} additional question types")
        
        # Filter out additional questions that duplicate QA space questions (by question text)
        qa_space_question_texts = set()
        for q in spatial_questions_from_qa_space + all_easy_questions_from_qa_space + non_spatial_questions_from_qa_space:
            question_text = q.get('question', '').strip()
            if question_text:
                qa_space_question_texts.add(question_text)
        
        filtered_additional_questions = []
        for q in additional_questions:
            question_text = q.get('question', '').strip()
            if question_text not in qa_space_question_texts:
                filtered_additional_questions.append(q)
            else:
                logger.debug(f"Skipping additional question that duplicates QA space question: {question_text[:50]}...")
        
        additional_questions = filtered_additional_questions
        logger.info(f"After filtering duplicates with QA space: {len(additional_questions)} additional questions")
        
        # Separate spatial vs non-spatial from additional questions
        spatial_from_additional = [q for q in additional_questions if q.get('question_type', '').startswith('spatial_')]
        easy_from_additional = self._filter_questions_by_type(additional_questions, self.EASY_QUESTION_TYPES)
        hard_from_additional = self._filter_questions_by_type(additional_questions, self.HARD_QUESTION_PREFIXES, use_prefix=True)
        # Non-spatial non-easy (shouldn't exist, but just in case)
        other_from_additional = [q for q in additional_questions 
                                if not q.get('question_type', '').startswith('spatial_')
                                and not any(q.get('question_type', '').startswith(prefix) for prefix in self.HARD_QUESTION_PREFIXES)
                                and q not in easy_from_additional]
        
        # Collect spatial questions for final cap (not added yet - will be handled in final cap with 60% chance)
        all_spatial_questions = spatial_questions_from_qa_space + spatial_from_additional
        if all_spatial_questions:
            logger.debug(f"Collected {len(all_spatial_questions)} spatial questions for final cap (QA space: {len(spatial_questions_from_qa_space)}, additional: {len(spatial_from_additional)})")
        else:
            logger.debug("No spatial questions generated (no valid spatial relationships found)")
        
        # Limit easy questions from additional to maintain easy_question_cap total per image
        # Count easy questions that were already selected from QA space
        current_easy_count = len([q for q in non_spatial_questions_from_qa_space 
                                  if self._is_easy_question(q)])
        remaining_easy_slots = max(0, easy_question_cap - current_easy_count)
        
        if easy_from_additional and remaining_easy_slots > 0:
            max_easy_additional = min(remaining_easy_slots, len(easy_from_additional))
            selected_easy_additional = self._select_easy_questions(easy_from_additional, max_easy_additional)
            non_spatial_questions_from_qa_space.extend(selected_easy_additional)
            logger.debug(
                "Selected %d additional easy questions (function=%d, material=%d, physical=%d, material_inference=%d, functional_seating=%d, description=%d, other=%d)",
                len(selected_easy_additional),
                sum(1 for q in selected_easy_additional if q.get('question_type') == 'function_knowledge'),
                sum(1 for q in selected_easy_additional if q.get('question_type') == 'material_property'),
                sum(1 for q in selected_easy_additional if q.get('question_type') == 'physical_property'),
                sum(1 for q in selected_easy_additional if q.get('question_type') in self.MATERIAL_INFERENCE_TYPES),
                sum(1 for q in selected_easy_additional if q.get('question_type') == 'functional_seating'),
                sum(1 for q in selected_easy_additional if q.get('question_type') == 'description_matching'),
                sum(1 for q in selected_easy_additional if q.get('question_type') not in {'function_knowledge', 'material_property', 'physical_property', 'functional_seating', 'description_matching'} | self.MATERIAL_INFERENCE_TYPES)
            )
        
        # Add all non-spatial questions (hard questions and limited easy questions)
        questions.extend(non_spatial_questions_from_qa_space)
        questions.extend(hard_from_additional)
        questions.extend(other_from_additional)

        # When generating sim-image questions, limit highly similar categories to reduce repetition
        if use_sm_valid_only:
            questions = self._apply_similarity_limits(questions)
        
        # Final cap: Limit total questions to max_questions_per_image with explicit priority:
        # Priority 1: Hard questions (always kept)
        # Priority 2: Easy questions (fill remaining slots, max easy_question_cap with function/material/physical priority)
        # Priority 3: Spatial questions (fill remaining slots after easy, 60% chance)
        total_questions = len(questions)
        if total_questions > max_questions_per_image:
            original_question_count = total_questions
            hard_question_types = ['repurposing_', 'counterfactual_', 'compositional_', 'latent_']
            hard_questions = [q for q in questions 
                            if any(q.get('question_type', '').startswith(hard_type) for hard_type in hard_question_types)]
            
            # Separate easy and spatial questions
            easy_questions = [q for q in questions 
                            if not any(q.get('question_type', '').startswith(hard_type) for hard_type in hard_question_types)
                            and not q.get('question_type', '').startswith('spatial_')]
            spatial_questions = all_spatial_questions  # Use collected spatial questions (not already in questions list)
            
            # Priority order: Hard > Easy (max easy_question_cap) > Spatial (60% chance)
            remaining_slots = max(0, max_questions_per_image - len(hard_questions))
            
            selected_easy: List[Dict[str, Any]] = []
            if easy_questions and remaining_slots > 0:
                max_easy = min(remaining_slots, easy_question_cap, len(easy_questions))
                if max_easy > 0:
                    selected_easy = self._select_easy_questions(easy_questions, max_easy)
                    remaining_slots -= len(selected_easy)
            
            # Fill remaining slots with spatial questions (60% chance)
            if spatial_questions and remaining_slots > 0:
                if random.random() < 0.6:  # 60% chance to include spatial
                    max_spatial = min(remaining_slots, len(spatial_questions))
                    selected_spatial = random.sample(spatial_questions, max_spatial)
                else:
                    selected_spatial = []
            else:
                selected_spatial = []
            
            questions = hard_questions + selected_easy + selected_spatial
            logger.info(f"Capped questions from {original_question_count} to {len(questions)} (kept {len(hard_questions)} hard, {len(selected_easy)} easy, {len(selected_spatial)} spatial)")
        else:
            # If under max_questions_per_image, still apply 60% chance for spatial questions
            remaining_slots = max(0, max_questions_per_image - total_questions)
            if all_spatial_questions and remaining_slots > 0:
                if random.random() < 0.6:  # 60% chance to include spatial
                    max_spatial = min(remaining_slots, len(all_spatial_questions))
                    selected_spatial = random.sample(all_spatial_questions, max_spatial)
                    questions.extend(selected_spatial)
                    logger.debug(f"Added {len(selected_spatial)} spatial question(s) to fill remaining slots (random 60% chance)")
        
        # Deduplicate: Replace exact duplicate questions with another question from the same category
        questions = self._deduplicate_questions_with_replacement(questions, all_spatial_questions)
        
        total_questions = len(questions)
        
        return questions
    
    def _deduplicate_questions_with_replacement(self, questions: List[Dict[str, Any]], 
                                                unused_spatial_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove exact duplicate questions and replace with another question from the same category"""
        seen_questions = {}  # image_id -> set of question texts
        deduplicated = []
        duplicates_replaced = 0
        
        # Separate questions by category for replacement pool
        hard_question_types = ['repurposing_', 'counterfactual_', 'compositional_', 'latent_']
        replacement_pools = {
            'hard': [],
            'easy': [],
            'spatial': []
        }
        
        # First pass: collect all questions and categorize them
        all_questions_by_category = {
            'hard': [],
            'easy': [],
            'spatial': []
        }
        
        for q in questions:
            question_type = q.get('question_type', '')
            if any(question_type.startswith(prefix) for prefix in hard_question_types):
                all_questions_by_category['hard'].append(q)
            elif question_type.startswith('spatial_'):
                all_questions_by_category['spatial'].append(q)
            else:
                all_questions_by_category['easy'].append(q)
        
        # Add unused spatial questions to replacement pool
        for q in unused_spatial_questions:
            question_text = q.get('question', '').strip()
            # Only add if not already in questions
            if not any(existing_q.get('question', '').strip() == question_text for existing_q in questions):
                replacement_pools['spatial'].append(q)
        
        # Second pass: deduplicate and replace
        for q in questions:
            image_id = q.get('image_id', 'unknown')
            question_text = q.get('question', '').strip()
            
            # Create a key for this image's questions
            if image_id not in seen_questions:
                seen_questions[image_id] = set()
            
            # Check if we've seen this exact question text for this image
            if question_text in seen_questions[image_id]:
                duplicates_replaced += 1
                logger.debug(f"Found duplicate question for {image_id}: {question_text[:50]}...")
                
                # Determine category and find replacement
                question_type = q.get('question_type', '')
                if any(question_type.startswith(prefix) for prefix in hard_question_types):
                    category = 'hard'
                elif question_type.startswith('spatial_'):
                    category = 'spatial'
                else:
                    category = 'easy'
                
                # Try to find a replacement from the same category
                replacement = None
                replacement_source = None
                
                # Check unused questions in same category first
                if replacement_pools[category]:
                    replacement = replacement_pools[category].pop(0)
                    replacement_source = 'replacement_pool'
                else:
                    # Check other questions in same category that haven't been added yet
                    available_replacements = [
                        rq for rq in all_questions_by_category[category]
                        if rq.get('question', '').strip() not in seen_questions.get(image_id, set())
                        and rq.get('question', '').strip() != question_text
                    ]
                    if available_replacements:
                        replacement = random.choice(available_replacements)
                        replacement_source = 'same_category'
                
                if replacement:
                    replacement_text = replacement.get('question', '').strip()
                    seen_questions[image_id].add(replacement_text)
                    deduplicated.append(replacement)
                    logger.debug(f"Replaced duplicate with {category} question from {replacement_source}: {replacement_text[:50]}...")
                else:
                    logger.debug(f"No replacement available for {category} category, skipping duplicate")
                continue
            
            # Add to seen set and keep the question
            seen_questions[image_id].add(question_text)
            deduplicated.append(q)
        
        if duplicates_replaced > 0:
            logger.info(f"Replaced {duplicates_replaced} duplicate questions with questions from the same category")
        
        return deduplicated

    def _select_easy_questions(self, easy_questions: List[Dict[str, Any]], max_easy: int) -> List[Dict[str, Any]]:
        """Select easy questions with priority ordering."""
        if not easy_questions or max_easy <= 0:
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for question in easy_questions:
            q_type = question.get('question_type', '')
            grouped.setdefault(q_type, []).append(question)

        selected: List[Dict[str, Any]] = []
        used_ids: Set[int] = set()

        def pick_type(q_type: str, limit: int | None) -> None:
            if len(selected) >= max_easy:
                return
            pool = grouped.get(q_type, [])
            available = [q for q in pool if id(q) not in used_ids]
            if not available:
                return
            allowed = limit if limit is not None else len(available)
            count = min(allowed, max_easy - len(selected), len(available))
            if count <= 0:
                return
            chosen = random.sample(available, count)
            selected.extend(chosen)
            used_ids.update(id(q) for q in chosen)

        def pick_from_pool(pool: List[Dict[str, Any]], limit: int | None) -> None:
            if len(selected) >= max_easy:
                return
            available = [q for q in pool if id(q) not in used_ids]
            if not available:
                return
            allowed = limit if limit is not None else len(available)
            count = min(allowed, max_easy - len(selected), len(available))
            if count <= 0:
                return
            chosen = random.sample(available, count)
            selected.extend(chosen)
            used_ids.update(id(q) for q in chosen)

        pick_type('function_knowledge', self.FUNCTION_EASY_TARGET)
        pick_type('material_property', self.MATERIAL_EASY_TARGET)
        pick_type('physical_property', self.PHYSICAL_EASY_TARGET)
        pick_type('functional_seating', None)

        if len(selected) < max_easy:
            other_pool = [
                q for q in easy_questions
                if id(q) not in used_ids
                and q.get('question_type') not in {
                    'function_knowledge',
                    'material_property',
                    'physical_property',
                    'functional_seating',
                    'description_matching'
                }
            ]
            pick_from_pool(other_pool, None)

        if len(selected) < max_easy and self.DESCRIPTION_EASY_LIMIT > 0:
            pick_type('description_matching', min(self.DESCRIPTION_EASY_LIMIT, max_easy - len(selected)))

        if len(selected) < max_easy:
            fallback_pool = [q for q in easy_questions if id(q) not in used_ids]
            pick_from_pool(fallback_pool, None)

        return selected

    def _apply_similarity_limits(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Limit similar question categories to reduce repetition on a single image."""
        if not questions:
            return questions

        limits = [
            {
                'name': 'container_like',
                'types': {'latent_containment', 'repurposing_container_concept', 'affordance_contain__carry__package'},
                'limit': 1
            },
            {
                'name': 'compactness_like',
                'types': {'latent_compressible', 'functional_foldable'},
                'limit': 1
            },
            {
                'name': 'repurposing',
                'prefix': 'repurposing_',
                'limit': 2
            },
            {
                'name': 'affordance',
                'prefix': 'affordance_',
                'limit': 2
            },
            {
                'name': 'rep_aff_combined',
                'types': {'_COMBINED_REP_AFF_'},
                'limit': 3
            },
            {
                'name': 'repurposing_stepstool',
                'types': {'repurposing_stepstool_concept'},
                'limit': 1
            },
            {
                'name': 'spatial',
                'prefix': 'spatial_',
                'limit': 2
            }
        ]

        indices_to_remove: Set[int] = set()

        for group in limits:
            limit = group['limit']
            if limit <= 0:
                continue

            matching_indices: List[int] = []
            for idx, question in enumerate(questions):
                if idx in indices_to_remove:
                    continue
                q_type = question.get('question_type', '')
                if group['name'] == 'rep_aff_combined':
                    if q_type.startswith('repurposing_') or q_type.startswith('affordance_'):
                        matching_indices.append(idx)
                    continue
                match = False
                if 'types' in group and q_type in group['types']:
                    match = True
                elif 'prefix' in group and q_type.startswith(group['prefix']):
                    match = True
                if match:
                    matching_indices.append(idx)

            if len(matching_indices) > limit:
                # Keep the first "limit" questions, remove the rest to minimize repetition
                for idx in matching_indices[limit:]:
                    indices_to_remove.add(idx)
                logger.info(
                    "Limiting %s questions from %d to %d",
                    group['name'],
                    len(matching_indices),
                    limit
                )

        if not indices_to_remove:
            return questions

        filtered_questions = [q for idx, q in enumerate(questions) if idx not in indices_to_remove]
        return filtered_questions
    
    def _is_object_in_void_cluster(self, target_object: str, taxonomy_type: str) -> bool:
        """Check if object is in a void cluster for the given taxonomy type"""
        try:
            if not self.taxonomy_utils:
                return False
            clusters = self.taxonomy_utils.get_object_clusters(target_object, f'final_taxonomy_{taxonomy_type}')
            if clusters:
                # Check if any of the object's clusters is a void cluster
                is_void = any(is_void_cluster(cluster, taxonomy_type) for cluster in clusters)
                if is_void:
                    return True
            
            # For material questions, also check affordance clusters (person/occupation exclusion)
            # Person/occupation objects should be excluded from material questions even if not in material void clusters
            if taxonomy_type == 'material':
                affordance_clusters = self.taxonomy_utils.get_object_clusters(target_object, 'final_taxonomy_affordances')
                if affordance_clusters and 'Human Roles & Identities (Occupations/Person Types)' in affordance_clusters:
                    return True
            
            return False
        except Exception:
            return False
    
    def _is_object_in_any_void_cluster(self, target_object: str) -> bool:
        """Check if object is in any relevant void cluster (affordance, shape, physical, texture) for description matching"""
        taxonomy_types_to_check = ['affordance', 'shape', 'physical', 'texture']
        for tax_type in taxonomy_types_to_check:
            if self._is_object_in_void_cluster(target_object, tax_type):
                return True
        return False
    
    def _has_description_conflict(self, target_object: str, other_objects: List[str]) -> bool:
        """Check if description_matching question would have ambiguous conflicts (Option 3: Hybrid approach)"""
        if not self.object_utils:
            return False
        
        # Get target object's key properties
        target_description = self.object_utils.get_object_description(target_object)
        if not target_description or not target_description.strip():
            return True  # Skip if no description
        
        target_material = self.object_utils.get_object_material(target_object)
        target_function = self.object_utils.get_object_function(target_object)
        
        # Check for conflicts with other objects
        conflicts_found = 0
        for other_obj in other_objects:
            other_description = self.object_utils.get_object_description(other_obj)
            if not other_description or not other_description.strip():
                continue
            
            other_material = self.object_utils.get_object_material(other_obj)
            other_function = self.object_utils.get_object_function(other_obj)
            
            # Conflict check 1: Same material AND same function → high conflict risk
            if target_material and other_material and target_material.strip() and other_material.strip():
                if target_material.strip().lower() == other_material.strip().lower():
                    if target_function and other_function and target_function.strip() and other_function.strip():
                        if target_function.strip().lower() == other_function.strip().lower():
                            conflicts_found += 1
                            logger.debug(f"Description conflict: {target_object} and {other_obj} share material '{target_material}' and function '{target_function}'")
                            continue
            
            # Conflict check 2: High text similarity between descriptions
            similarity = self._calculate_text_similarity(target_description, other_description)
            if similarity > 0.7:  # 70% similarity threshold
                conflicts_found += 1
                logger.debug(f"Description conflict: {target_object} and {other_obj} have high text similarity ({similarity:.2f})")
        
        # If 2+ conflicts found, consider it ambiguous
        return conflicts_found >= 1  # Skip if even 1 conflict (conservative approach)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap (0.0 to 1.0)"""
        if not text1 or not text2:
            return 0.0
        
        # Normalize text
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _is_high_reliability_orientation(self, class_name: str) -> bool:
        """Check if object class typically has reliable orientation (front/back/left/right)"""
        if not class_name:
            return False
        return class_name.lower() in HIGH_RELIABILITY_CLASSES
    
    def _get_class_name_mapping(self, image_id: str, object_names: List[str]) -> Dict[str, str]:
        """
        Get mapping from object_name to class_name for given objects
        Caches the mapping per image_id to avoid repeated annotation loads
        """
        if not hasattr(self, '_class_name_cache'):
            self._class_name_cache = {}
        
        # Check cache first
        cache_key = image_id
        if cache_key in self._class_name_cache:
            cached_map = self._class_name_cache[cache_key]
            # Return only requested objects
            return {obj_name: cached_map.get(obj_name, '') for obj_name in object_names}
        
        # Load annotations
        if not self.annotations_dir:
            return {obj_name: '' for obj_name in object_names}
        
        image_dir = self.annotations_dir / image_id
        annotation_files = list(image_dir.glob("annotations/*.json"))
        
        if not annotation_files:
            return {obj_name: '' for obj_name in object_names}
        
        try:
            with open(annotation_files[0], 'r') as f:
                annotations = json.load(f)
            
            # Build mapping for all objects in image
            class_name_map = {}
            for detection in annotations.get('detections', []):
                obj_name = detection.get('class_name', '')
                # Use class_name as both key and value (object_name == class_name in annotations)
                class_name_map[obj_name] = obj_name
            
            # Cache the full mapping
            self._class_name_cache[cache_key] = class_name_map
            
            # Return only requested objects
            return {obj_name: class_name_map.get(obj_name, '') for obj_name in object_names}
        except Exception as e:
            logger.warning(f"Error loading class name mapping for {image_id}: {e}")
            return {obj_name: '' for obj_name in object_names}
    
    def _get_other_objects(self, available_objects: List[str], target_object: str) -> List[str]:
        """Helper method to get all objects except the target object"""
        return [obj for obj in available_objects if obj != target_object]
    
    def _is_taxonomy_based_question(self, question_type: str) -> bool:
        """Check if question type requires taxonomy cluster conflict checking"""
        taxonomy_based_types = ['function_knowledge', 'functional_seating', 'functional_foldable', 'physical_property']
        return (question_type in taxonomy_based_types or 
                question_type.startswith('affordance_') or
                (question_type.startswith('material_') and question_type != 'material_property'))
    
    def _get_description_source(self, target_object: str) -> Tuple[str, str]:
        """
        Get description and determine its source taxonomy type.
        
        Returns:
            (description, source_type) where source_type is one of:
            - 'general' - from 'description' field (general description)
            - 'material' - from 'material' field
            - 'function' - from 'functions' field
            - 'texture' - from 'texture' field (if exists)
            - 'affordance' - from 'affordance' field (if exists)
            - None if no description found
        """
        if not self.object_utils or not self.object_utils.object_descriptions:
            return (None, None)
        
        obj_data = self.object_utils.object_descriptions.get(target_object)
        if not obj_data:
            return (None, None)
        
        # Check general description first
        description = obj_data.get('description', '')
        if description:
            cleaned = self.object_utils._clean_malformed_text(description) if hasattr(self.object_utils, '_clean_malformed_text') else description.strip()
            if cleaned:
                return (cleaned, 'general')
        
        # Check material
        material = obj_data.get('material', [])
        if material and isinstance(material, list) and len(material) > 0:
            if isinstance(material[0], str):
                material_text = self.object_utils._clean_malformed_text(material[0]) if hasattr(self.object_utils, '_clean_malformed_text') else material[0].strip()
                if material_text:
                    return (f"made of {material_text}", 'material')
        
        # Check function
        functions = obj_data.get('functions', [])
        if functions and isinstance(functions, list) and len(functions) > 0:
            if isinstance(functions[0], str):
                function_text = self.object_utils._clean_malformed_text(functions[0]) if hasattr(self.object_utils, '_clean_malformed_text') else functions[0].strip()
                if function_text:
                    return (f"used for {function_text}", 'function')
        
        # Check texture (if exists)
        texture = obj_data.get('texture', [])
        if texture and isinstance(texture, list) and len(texture) > 0:
            if isinstance(texture[0], str):
                texture_text = self.object_utils._clean_malformed_text(texture[0]) if hasattr(self.object_utils, '_clean_malformed_text') else texture[0].strip()
                if texture_text:
                    return (f"has {texture_text} texture", 'texture')
        
        # Check affordance (if exists)
        affordance = obj_data.get('affordance', [])
        if affordance and isinstance(affordance, list) and len(affordance) > 0:
            if isinstance(affordance[0], str):
                affordance_text = self.object_utils._clean_malformed_text(affordance[0]) if hasattr(self.object_utils, '_clean_malformed_text') else affordance[0].strip()
                if affordance_text:
                    return (f"can {affordance_text}", 'affordance')
        
        return (None, None)
    
    def _generate_question_text_for_type(self, question_type: str, target_object: str, 
                                        available_objects: List[str]) -> str:
        """Generate question text for a specific question type with validation"""
        if question_type == 'description_matching':
            if self._is_object_in_any_void_cluster(target_object):
                logger.debug(f"Skipping description_matching for {target_object}: object is in void cluster")
                return None
            
            # Get description and its source taxonomy type
            description, source_type = self._get_description_source(target_object)
            if not description or not description.strip():
                return None
            
            # For taxonomy-based descriptions, check cluster conflicts
            taxonomy_based_sources = ['material', 'function', 'texture', 'affordance']
            if source_type in taxonomy_based_sources:
                # Map source type to taxonomy file name
                taxonomy_mapping = {
                    'material': 'final_taxonomy_material',
                    'function': 'final_taxonomy_function',
                    'texture': 'final_taxonomy_texture',
                    'affordance': 'final_taxonomy_affordances'
                }
                taxonomy_file = taxonomy_mapping.get(source_type)
                
                if taxonomy_file:
                    target_clusters = self.taxonomy_utils.get_object_clusters(target_object, taxonomy_file)
                    if target_clusters:
                        # Check if other objects share the same cluster
                        other_objects = self._get_other_objects(available_objects, target_object)
                        conflicting_objects = []
                        for obj in other_objects:
                            obj_clusters = self.taxonomy_utils.get_object_clusters(obj, taxonomy_file)
                            if obj_clusters and any(cluster in obj_clusters for cluster in target_clusters):
                                conflicting_objects.append(obj)
                        
                        if conflicting_objects:
                            logger.debug(f"Skipping description_matching for {target_object}: conflicts with {len(conflicting_objects)} object(s) "
                                      f"in same {source_type} cluster(s) {target_clusters}: {conflicting_objects}")
                            return None
            
            # Also check for general description conflicts (existing logic)
            other_objects = self._get_other_objects(available_objects, target_object)
            if self._has_description_conflict(target_object, other_objects):
                logger.debug(f"Skipping description_matching for {target_object}: description conflicts")
                return None
            
            if description and description.strip():
                return get_question_template(question_type, description=description)
            return None
        
        elif question_type == 'function_knowledge':
            if self._is_object_in_void_cluster(target_object, 'function'):
                return None
            
            function = self.object_utils.get_object_function(target_object)
            if not function or not function.strip():
                return None
            
            if self._has_taxonomy_cluster_conflict(target_object, available_objects, question_type):
                logger.debug(f"Skipping {question_type} for {target_object}: taxonomy cluster conflict")
                return None
            
            return get_question_template(question_type, function=function)
        
        elif question_type == 'material_property':
            if self._is_object_in_void_cluster(target_object, 'material'):
                return None
            
            material = self.object_utils.get_object_material(target_object)
            if not material or not material.strip() or len(material) > 40:
                return None
            
            if not self._is_material_unique(target_object, available_objects, material):
                return None
            
            if self._has_taxonomy_cluster_conflict(target_object, available_objects, question_type):
                logger.debug(f"Skipping {question_type} for {target_object}: taxonomy cluster conflict")
                return None
            
            return get_question_template(question_type, material=material)
        
        return None
    
    def _is_material_unique(self, target_object: str, available_objects: List[str], target_material: str = None) -> bool:
        """
        Check if only ONE object in the scene has the target material (to avoid ambiguity).
        
        Args:
            target_object: The object to check material uniqueness for
            available_objects: All objects in the scene
            target_material: Optional material string. If not provided, will be fetched from object_utils.
        
        Returns:
            True if target_object is the ONLY one with this material, False otherwise.
        """
        if target_material is None:
            target_material = self.object_utils.get_object_material(target_object)
        
        if not target_material or not target_material.strip():
            return True  # If no material, consider it unique (won't generate question anyway)
        
        other_objects_in_scene = self._get_other_objects(available_objects, target_object)
        objects_with_same_material = [target_object]
        
        for other_obj in other_objects_in_scene:
            other_material = self.object_utils.get_object_material(other_obj)
            if other_material and other_material.strip():
                if other_material.strip().lower() == target_material.strip().lower():
                    objects_with_same_material.append(other_obj)
        
        # Only generate if target_object is the ONLY one with this material
        if len(objects_with_same_material) > 1:
            logger.debug(f"Skipping material_property for {target_object}: material '{target_material}' is shared by {len(objects_with_same_material)} objects: {objects_with_same_material}")
            return False
        
        return True
    
    def _has_taxonomy_cluster_conflict(self, target_object: str, available_objects: List[str], question_type: str) -> bool:
        """
        Check if there are conflicts in taxonomy clusters for taxonomy-specific questions.
        Returns True if conflicts exist (meaning question should be skipped).
        
        Args:
            target_object: The target object for the question
            available_objects: All objects in the scene (including target)
            question_type: Type of question being generated
            
        Returns:
            True if conflicts found (skip question), False if no conflicts (generate question)
        """
        other_objects = self._get_other_objects(available_objects, target_object)
        
        # Special case: Cross-cluster conflict detection for related infrastructure clusters
        if question_type == 'affordance_enclosures_and_venues_(enter_use)':
            # Exclude objects in "Build / Span / Occupy" cluster (similar large infrastructure)
            for obj in other_objects:
                obj_clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_affordances')
                if obj_clusters and 'Build / Span / Occupy' in obj_clusters:
                    logger.debug(f"Skipping {question_type} for {target_object}: conflicts with {obj} "
                               f"in 'Build / Span / Occupy' cluster")
                    return True
        
        elif question_type == 'affordance_build__span__occupy':
            # Exclude objects in "Enclosures & Venues (Enter/Use)" cluster (similar large infrastructure)
            for obj in other_objects:
                obj_clusters = self.taxonomy_utils.get_object_clusters(obj, 'final_taxonomy_affordances')
                if obj_clusters and 'Enclosures & Venues (Enter/Use)' in obj_clusters:
                    logger.debug(f"Skipping {question_type} for {target_object}: conflicts with {obj} "
                               f"in 'Enclosures & Venues (Enter/Use)' cluster")
                    return True
        
        # Standard same-cluster conflict detection for other question types
        # Get taxonomy type and relevant clusters for this question type
        taxonomy_type, target_clusters = self._get_taxonomy_cluster_for_question(target_object, question_type)
        
        if not target_clusters:
            # If we can't determine clusters for the target object, we can't verify there's no conflict
            # This could happen if:
            # 1. Object is not in taxonomy (e.g., misspelled like "cabinate" instead of "cabinet")
            # 2. Object doesn't have the relevant cluster
            # To be safe, we should check if OTHER objects in the scene have clusters for this question type
            # If they do, and we can't verify the target object's clusters, skip to avoid ambiguity
            for obj in other_objects:
                obj_taxonomy_type, obj_clusters = self._get_taxonomy_cluster_for_question(obj, question_type)
                if obj_clusters:
                    # Another object has clusters for this question type, but target doesn't
                    # This is ambiguous - skip the question
                    logger.debug(f"Skipping {question_type} for {target_object}: target object not found in taxonomy "
                               f"but other object {obj} has clusters {obj_clusters}")
                    return True
            # If no other objects have clusters either, allow question (all objects missing from taxonomy)
            return False
        
        # Check if any other object shares the same cluster(s)
        conflicting_objects = []
        for obj in other_objects:
            obj_clusters = self.taxonomy_utils.get_object_clusters(obj, taxonomy_type)
            
            # Check if object shares ANY cluster with target
            shares_cluster = any(cluster in obj_clusters for cluster in target_clusters)
            
            if shares_cluster:
                conflicting_objects.append(obj)
        
        if conflicting_objects:
            if question_type == 'function_knowledge' and self.object_utils:
                target_function = (self.object_utils.get_object_function(target_object) or "").strip().lower()
                if target_function:
                    unresolved_conflicts = []
                    for obj in conflicting_objects:
                        obj_function = (self.object_utils.get_object_function(obj) or "").strip().lower()
                        if not obj_function or obj_function == target_function:
                            unresolved_conflicts.append(obj)
                    if unresolved_conflicts:
                        logger.debug(
                            f"Skipping {question_type} for {target_object}: conflicts with {len(unresolved_conflicts)} object(s) "
                            f"sharing function '{target_function}' or missing function metadata: {unresolved_conflicts}"
                        )
                        return True
                    # All conflicting objects have distinct functions – treat as safe
                    return False
            logger.debug(f"Skipping {question_type} for {target_object}: conflicts with {len(conflicting_objects)} object(s) "
                        f"in same cluster(s) {target_clusters}: {conflicting_objects}")
            return True
        
        return False
    
    def _get_taxonomy_cluster_for_question(self, target_object: str, question_type: str):
        """
        Get taxonomy type and relevant cluster(s) for a question type.
        
        Returns:
            (taxonomy_type, list_of_cluster_names) or (None, []) if cannot determine
        """
        if question_type == 'function_knowledge':
            if not self.object_utils:
                return (None, [])
            function = self.object_utils.get_object_function(target_object)
            if function and function.strip():
                clusters = self.taxonomy_utils.get_object_clusters(target_object, 'final_taxonomy_function')
                return ('final_taxonomy_function', clusters)
        
        elif question_type == 'material_property':
            # Already handled by _is_material_unique, but include for completeness
            if not self.object_utils:
                return (None, [])
            material = self.object_utils.get_object_material(target_object)
            if material and material.strip():
                clusters = self.taxonomy_utils.get_object_clusters(target_object, 'final_taxonomy_material')
                return ('final_taxonomy_material', clusters)
        
        elif question_type == 'physical_property':
            if not self.object_utils:
                return (None, [])
            properties = self.object_utils.get_object_physical_properties(target_object)
            if properties:
                clusters = self.taxonomy_utils.get_object_clusters(target_object, 'final_taxonomy_physical_properties')
                return ('final_taxonomy_physical_properties', clusters)
        
        elif question_type.startswith('affordance_'):
            clusters = self.taxonomy_utils.get_object_clusters(target_object, 'final_taxonomy_affordances')
            if not clusters:
                return (None, [])
            
            # Find matching cluster using mapping
            expected_cluster_name = self.AFFORDANCE_CLUSTER_MAP.get(question_type)
            if expected_cluster_name:
                matching_clusters = [c for c in clusters if expected_cluster_name in c or c in expected_cluster_name]
                return ('final_taxonomy_affordances', matching_clusters if matching_clusters else clusters)
            return ('final_taxonomy_affordances', clusters)
        
        elif question_type.startswith('material_') and question_type != 'material_property':
            clusters = self.taxonomy_utils.get_object_clusters(target_object, 'final_taxonomy_material')
            if not clusters:
                return (None, [])
            
            # Find matching cluster using mapping
            expected_cluster_name = self.MATERIAL_CLUSTER_MAP.get(question_type)
            if expected_cluster_name:
                matching_clusters = [c for c in clusters if expected_cluster_name in c or c in expected_cluster_name]
                return ('final_taxonomy_material', matching_clusters if matching_clusters else clusters)
            return ('final_taxonomy_material', clusters)
        
        elif question_type == 'functional_seating':
            clusters = self.taxonomy_utils.get_object_clusters(target_object, 'final_taxonomy_affordances')
            if not clusters:
                return (None, [])
            matching = [c for c in clusters if 'Sit / Ride / Attend' in c]
            if matching:
                return ('final_taxonomy_affordances', matching)
            return (None, [])
        
        elif question_type == 'functional_foldable':
            clusters = self.taxonomy_utils.get_object_clusters(target_object, 'final_taxonomy_material')
            if not clusters:
                return (None, [])
            # Check for foldable material clusters
            foldable_clusters = [
                'Textiles, Fibers & Leather',
                'Plastics, Rubber & Polymers'
            ]
            matching = [c for c in clusters if c in foldable_clusters]
            if matching:
                return ('final_taxonomy_material', matching)
            # Also check if object is in Flexible physical property
            if self.taxonomy_utils.has_property(target_object, 'flexible'):
                return ('final_taxonomy_material', clusters)
            return (None, [])
        
        return (None, [])
    
    def _generate_answer_and_reasoning(self, question_type: str, target_object: str, 
                                       available_objects: List[str], reference_object: str = None,
                                       spatial_relationship: Dict = None, description: str = None) -> Tuple[str, str]:
        """Generate answer and reasoning for a question"""
        cot_generator = CoTReasoningGenerator(taxonomy_utils=self.taxonomy_utils)
        
        if question_type.startswith('spatial_'):
            if spatial_relationship and reference_object:
                answer = self.object_utils.get_spatial_answer(
                    question_type, target_object, reference_object, spatial_relationship
                )
                # Skip if answer is unknown (calculation determined it's ambiguous)
                if answer == 'unknown':
                    return (None, None)
                
                reasoning = cot_generator.generate_comprehensive_reasoning(
                    question_type, target_object, available_objects, answer,
                    spatial_context={'relationship': spatial_relationship}
                )
                return (answer, reasoning)
            return (None, None)
        
        answer = target_object
        if question_type == 'description_matching' and description:
            reasoning = cot_generator.generate_comprehensive_reasoning(
                question_type, target_object, available_objects, answer, description=description
            )
        else:
            reasoning = cot_generator.generate_comprehensive_reasoning(
                question_type, target_object, available_objects, answer
            )
        return (answer, reasoning)
    
    def _is_easy_question(self, question: Dict) -> bool:
        """Check if a question is an easy question type
        
        Note: question_type should be present in code logic (it's only removed 
        later when converting to JSON output as question_category)
        """
        question_type = question.get('question_type', '')
        
        # question_type should be available at this point in the code
        # It only gets converted to question_category later in generate_taxonomyqabench_realimage.py
        if question_type:
            return any(easy_type in question_type or question_type.startswith(easy_type) 
                      for easy_type in self.EASY_QUESTION_TYPES)
        
        # Fallback: if question_type is missing (shouldn't happen), check question_category
        # This can happen if questions are passed after JSON conversion
        question_category = question.get('question_category', '')
        if question_category:
            easy_categories = ['description', 'function', 'material', 'affordance', 'capability']
            return question_category in easy_categories
        
        return False
    
    def _filter_questions_by_type(self, questions: List[Dict], type_patterns: List[str], use_prefix: bool = False) -> List[Dict]:
        """
        Filter questions by type patterns.
        
        Args:
            questions: List of question dictionaries
            type_patterns: List of question type patterns to match
            use_prefix: If True, match if question type starts with any pattern. If False, match if pattern is in question type.
        
        Returns:
            Filtered list of questions
        """
        if use_prefix:
            return [q for q in questions 
                   if any(q.get('question_type', '').startswith(prefix) for prefix in type_patterns)]
        else:
            return [q for q in questions 
                   if any(pattern in q.get('question_type', '') for pattern in type_patterns)]
    
    def generate_additional_questions_not_in_qa_space(self, available_objects: List[str], image_id: str) -> List[Dict[str, Any]]:
        """Generate additional questions for question types not in QA space"""
        additional_questions = []
        
        # Question types not in QA space that we can generate directly
        # NOTE: function_knowledge and material_property are in QA space, so they should NOT be here
        # Only include question types that are truly NOT in QA space analysis
        # Excluding spatial_distance as it requires proper distance calculation implementation
        non_qa_question_types = [
            'description_matching',  # Not in QA space
            'spatial_above_below',   # Format: "Is X above or below Y?" (binary choice) - Not in QA space
            'spatial_left_right',    # Format: "Is X to the left or right of Y?" (binary choice) - Not in QA space
            'spatial_front_behind',  # Format: "Is X in front of or behind Y?" (binary choice) - Not in QA space
            # 'spatial_distance' excluded - requires complex distance calculations
            # 'function_knowledge' removed - already in QA space
            # 'material_property' removed - already in QA space
        ]
        
        # Generate 1-2 questions for each non-QA question type
        # Note: Spatial questions will be limited globally in generate_qa_from_space
        seen_additional_question_texts = set()  # Track additional questions to prevent duplicates
        for question_type in non_qa_question_types:
            if len(available_objects) >= 2:  # Need at least 2 objects for meaningful questions
                num_questions = random.randint(1, 2)
                type_questions = []
                seen_question_texts = set()  # Track generated questions to avoid duplicates within this type
                
                for _ in range(num_questions):
                    target_object = random.choice(available_objects)
                    
                    # Skip objects in void clusters for question-type-specific checks
                    if question_type == 'description_matching':
                        if self._is_object_in_any_void_cluster(target_object):
                            continue
                    
                    question_text = None
                    spatial_relationship = None
                    reference_object = None
                    description = None
                    
                    if question_type.startswith('spatial_'):
                        other_objects = self._get_other_objects(available_objects, target_object)
                        if not other_objects:
                            continue
                        
                        reference_object = random.choice(other_objects)
                        
                        # Filter for left/right and front/behind: require at least one high-reliability object
                        if question_type in ['spatial_left_right', 'spatial_front_behind']:
                            # Get class names for both objects
                            class_name_map = self._get_class_name_mapping(image_id, [target_object, reference_object])
                            obj1_class = class_name_map.get(target_object, '')
                            obj2_class = class_name_map.get(reference_object, '')
                            
                            # Check if at least one is high-reliability
                            if not (self._is_high_reliability_orientation(obj1_class) or 
                                    self._is_high_reliability_orientation(obj2_class)):
                                continue  # Skip: both are low-reliability
                        
                        spatial_relationship = self._get_spatial_relationship_for_question_type(
                            question_type, target_object, reference_object, image_id
                        )
                        
                        if spatial_relationship is None:
                            continue
                        
                        if question_type in ['spatial_left_right', 'spatial_above_below', 'spatial_front_behind']:
                            question_text = get_question_template(question_type, object1=target_object, object2=reference_object)
                        else:
                            question_text = get_question_template(question_type)
                    else:
                        question_text = self._generate_question_text_for_type(
                            question_type, target_object, available_objects
                        )
                        if question_type == 'description_matching':
                            description, _ = self._get_description_source(target_object)
                        if not question_text:
                            continue
                    
                    # Add options for multiple choice questions (non-spatial only)
                    if not question_type.startswith('spatial_'):
                        other_objects = self._get_other_objects(available_objects, target_object)
                        options = [target_object] + other_objects
                        question_text = f"{question_text} Objects to choose from: {', '.join(options)}"
                    
                    # Generate answer and reasoning
                    answer, reasoning = self._generate_answer_and_reasoning(
                        question_type, target_object, available_objects, 
                        reference_object, spatial_relationship, description
                    )
                    if answer is None:
                        continue
                    
                    question_obj = {
                        'question': question_text,
                        'answer': answer,
                        'question_type': question_type,
                        'target_object': target_object,
                        'objects': available_objects,
                        'choices': available_objects,  # Explicitly set choices to all available objects
                        'reasoning': reasoning,
                        'question_index': len(type_questions),
                        'image_path': f"{image_id}/bbox.jpg",
                        'image_id': image_id
                    }
                    
                    # Check for duplicate question text before adding (both within type and across all additional questions)
                    if question_text not in seen_question_texts and question_text not in seen_additional_question_texts:
                        seen_question_texts.add(question_text)
                        seen_additional_question_texts.add(question_text)
                        type_questions.append(question_obj)
                    else:
                        logger.debug(f"Skipping duplicate additional question for {image_id}: {question_text[:50]}...")
                
                if type_questions:
                    additional_questions.extend(type_questions)
        
        return additional_questions
    
    def _get_spatial_relationship_for_question_type(self, question_type: str, target_object: str, reference_object: str, image_id: str) -> Dict[str, str]:
        """Get spatial relationship for real images using annotations"""
        if self.annotations_dir:
            # Use object_utils to calculate spatial relationship from annotations
            spatial_relationship = self.object_utils.get_spatial_relationship(
                target_object, reference_object, image_id, annotations_dir=self.annotations_dir
            )
            return spatial_relationship
        else:
            # Fallback for sim images or when no annotations_dir is provided
            return {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown"}
    
    def generate_questions_for_objects_sim(self, objects: List[str], scene_id: str, object_poses: Dict[str, Dict[str, Any]], scene_path: Path = None) -> List[Dict[str, Any]]:
        """Generate questions for sim image objects with pose data"""
        questions = []
        
        # Initialize CoT reasoning generator
        cot_generator = CoTReasoningGenerator(taxonomy_utils=self.taxonomy_utils)
        
        # Define question types to generate for sim images
        # Generate ALL question types from QA space - let validation filter out what doesn't have valid answers
        question_types = [
            # Material questions
            'material_textiles_fibers_and_leather', 'material_metals_and_alloys', 'material_wood_and_plant_based_solids',
            'material_plastics_rubber_and_polymers', 'material_paper_cardboard_and_pulp', 'material_glass_and_transparent_(silicate)',
            'material_biological_(plants/flowers)', 'material_organic_food_and_edible_matter', 'material_biological_(animals/body_parts)',
            'material_ceramics_porcelain_and_earthenware', 'material_composites_and_multi_material_products', 'material_stone_concrete_and_mineral',
            'material_liquids_and_semi_liquids', 'material_gases_vapors_and_atmospheric', 'material_property',
            'material_sound_absorption', 'material_thermal_touch', 'material_scratch_resistance',
            # Affordance questions
            'affordance_furniture', 'affordance_contain__carry__package', 'affordance_grip__carry__operate',
            'affordance_operate__use_device', 'affordance_mechanical_control', 'affordance_mediated_action_and_meaning', 'affordance_food_—_ingredients_and_produce',
            'affordance_food_—_prepared_dishes', 'affordance_cleaning_and_sanitation', 'affordance_control__express__light',
            'affordance_grow__plant_(vegetation)', 'affordance_architectural_components_and_fixtures', 'affordance_art_display_(view_appraise)',
            'affordance_build__span__occupy', 'affordance_display__exhibit__signal_value', 'affordance_enclosures_and_venues_(enter_use)',
            'affordance_household__facility_operations', 'affordance_interact_with_living_moving_things', 'affordance_place__support__work_on',
            'affordance_structured_operational_engagement', 'affordance_tableware_and_serveware',
            'affordance_wearables_and_apparel',
            # Functional questions
            'functional_seating', 'functional_foldable', 'flammability',
            # Latent state questions
            'latent_containment', 'latent_compressible',
            # Compositional questions
            'compositional_set_subtraction_container', 'compositional_set_subtraction_hollow',
            # Counterfactual questions
            'counterfactual_water', 'counterfactual_heat',
            # Repurposing questions
            'repurposing_shield_concept', 'repurposing_bookend_concept', 'repurposing_container_concept',
            'repurposing_cushion_concept', 'repurposing_reflector_concept',
            'repurposing_stepstool_concept',
            # Spatial questions (only binary "Is X ... Y?" format)
            'spatial_above_below', 'spatial_left_right', 'spatial_front_behind', 'spatial_closer_to_camera',
            # Other
            'description_matching', 'function_knowledge'
        ]
        
        # Load QA space data once
        qa_space_data = self.data_loading_utils.load_qa_space_data()
        if qa_space_data:
            question_answer_mappings = qa_space_data.get('question_answer_mappings', {})
        else:
            question_answer_mappings = {}
        
        # Loop through EACH OBJECT and check for non-spatial questions it can answer
        for target_object in objects:
            # Skip objects in void clusters for question-type-specific checks
            # Check function void cluster for function_knowledge questions
            is_in_function_void = self._is_object_in_void_cluster(target_object, 'function') if self.taxonomy_utils else False
            # Check material void cluster for material questions
            is_in_material_void = self._is_object_in_void_cluster(target_object, 'material') if self.taxonomy_utils else False
            # Check any void cluster for description_matching
            is_in_any_void = self._is_object_in_any_void_cluster(target_object) if self.taxonomy_utils else False
            
            # For each object, check which question types it can answer
            for question_type in question_types:
                try:
                    if question_type in self.DISABLED_QUESTION_TYPES:
                        continue
                    # Skip spatial questions here - handle them separately
                    if question_type.startswith('spatial_'):
                        continue
                    
                    question_data_map = question_answer_mappings.get(question_type, {})
                    if not question_data_map:
                        continue
                    
                    # Get valid objects for this question type
                    valid_objects = set(question_data_map.get('sm_valid_objects', []) + question_data_map.get('openimages_valid_objects', []))
                    
                    # Check if target_object can answer this question type
                    if target_object not in valid_objects:
                        continue
                    
                    # Skip void cluster objects for specific question types
                    if question_type == 'function_knowledge' and is_in_function_void:
                        continue
                    if question_type.startswith('material_') and is_in_material_void:
                        continue
                    if question_type == 'description_matching' and is_in_any_void:
                        continue
                    
                    # Check for conflicts: Are there other objects in the group that can also answer?
                    other_objects = [obj for obj in objects if obj != target_object]
                    conflicting_objects = [obj for obj in other_objects if obj in valid_objects]
                    
                    # Additional taxonomy cluster conflict check for taxonomy-specific questions
                    has_taxonomy_conflict = False
                    if self._is_taxonomy_based_question(question_type):
                        has_taxonomy_conflict = self._has_taxonomy_cluster_conflict(target_object, objects, question_type)
                    
                    # Only generate if NO CONFLICTS (both QA space conflicts and taxonomy cluster conflicts)
                    if len(conflicting_objects) == 0 and not has_taxonomy_conflict:
                        # Generate question for this specific object
                        template_questions = self.question_generation_utils.generate_questions_for_type_wrapper(
                            question_type, [target_object], objects, scene_id, cot_generator, object_poses, scene_path
                        )
                        questions.extend(template_questions)
                        
                except Exception as e:
                    logger.warning(f"Error generating question {question_type} for {target_object}: {e}")
                    continue
        
        # TEMPORARILY DISABLED: Handle spatial questions separately (use all objects, no conflict checking)
        # Spatial questions require 3D pose data which is not ready yet
        # spatial_question_types = [qt for qt in question_types if qt.startswith('spatial_')]
        # for question_type in spatial_question_types:
        #     try:
        #         if len(objects) < 2:
        #             continue
        #             
        #         # For spatial questions, use all objects
        #         template_questions = self.question_generation_utils.generate_questions_for_type_wrapper(
        #             question_type, objects, objects, scene_id, cot_generator, object_poses, scene_path
        #         )
        #         questions.extend(template_questions)
        #         
        #     except Exception as e:
        #         logger.warning(f"Error generating spatial question {question_type}: {e}")
        #         continue
        
        logger.debug(f"Skipping spatial question generation for sim images (3D pose data not ready)")
        
        return questions
