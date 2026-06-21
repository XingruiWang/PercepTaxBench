import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

from modules.qa_modules.question_templates import get_question_template
from modules.qa_modules.cot_reasoning_utils import CoTReasoningGenerator

logger = logging.getLogger(__name__)

class QuestionGenerationUtils:
    """Utility class for generating questions with common logic"""
    
    def __init__(self, taxonomy_utils, object_utils=None):
        self.taxonomy_utils = taxonomy_utils
        self.object_utils = object_utils
    
    def generate_questions_for_type_wrapper(self, question_type: str, matching_objects: List[str],
                                            available_objects: List[str], scene_id: str, 
                                            cot_generator, object_poses: Dict = None, scene_path: Path = None) -> List[Dict[str, Any]]:
        """Wrapper for sim image question generation"""
        # For spatial questions, we need to generate them without requiring spatial_relationship
        if question_type.startswith('spatial_'):
            return self.generate_spatial_questions_sim(
                question_type, matching_objects, scene_id, cot_generator, object_poses, scene_path
            )
        
        # For non-spatial questions, use standard method
        question_data = {}
        return self.generate_questions_for_type(
            question_type, matching_objects, available_objects, question_data, scene_id
        )
    
    def clean_description_punctuation(self, text: str) -> str:
        """Remove specific punctuation from description text for better readability"""
        import re
        if not text:
            return ""
        
        # Replace specific punctuation patterns
        cleaned = text
        
        # Replace & with 'and'
        cleaned = cleaned.replace('&', 'and')
        
        # Replace / with ', '
        cleaned = cleaned.replace('/', ', ')
        
        # Replace ‑ with -
        cleaned = cleaned.replace('‑', '-')
        
        # Remove parentheses and their contents
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        
        # Remove square brackets and their contents
        cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
        
        # Remove extra punctuation marks but keep basic punctuation
        cleaned = re.sub(r'[^\w\s\.,!?-]', '', cleaned)
        
        # Remove extra spaces and clean up
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove leading/trailing punctuation
        cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned)
        
        return cleaned
    
    def generate_questions_for_type(self, question_type: str, matching_objects: List[str], 
                                   available_objects: List[str], question_data: Dict[str, Any], 
                                   image_id: str, spatial_relationship: Dict[str, str] = None,
                                   reference_object: str = None) -> List[Dict[str, Any]]:
        """Generate questions for a specific question type"""
        questions = []
        
        # Handle spatial questions differently
        if question_type.startswith('spatial_'):
            return self._generate_spatial_questions(
                question_type, matching_objects, available_objects, question_data, 
                image_id, spatial_relationship, reference_object
            )
        
        # For non-spatial questions, generate standard questions
        # Pre-fetch and validate required data for questions that need it
        material = None
        physical_property = None
        function = None
        description = None
        
        # Validate material_property and function_knowledge questions before generating
        if question_type == 'material_property':
            from modules.qa_modules.filter_utils import is_void_cluster
            if self.object_utils and self.object_utils.taxonomy_utils:
                if self.object_utils.taxonomy_utils:
                    clusters = self.object_utils.taxonomy_utils.get_object_clusters(matching_objects[0], 'final_taxonomy_material')
                    if clusters and any(is_void_cluster(cluster, 'material') for cluster in clusters):
                        return questions
                    # Exclude person/occupation objects from material questions
                    affordance_clusters = self.object_utils.taxonomy_utils.get_object_clusters(matching_objects[0], 'final_taxonomy_affordances')
                    if 'Human Roles & Identities (Occupations/Person Types)' in affordance_clusters:
                        logger.debug(f"Skipping material_property for {matching_objects[0]}: object is a person/occupation")
                        return questions
            # Check if material data exists
            if not self.object_utils:
                return questions
            material = self.object_utils.get_object_material(matching_objects[0])
            # Material is already cleaned and validated by get_object_material
            # Additional check: reject if material is empty or too long
            if not material or not material.strip() or len(material) > 40:
                return questions
        elif question_type == 'physical_property':
            from modules.qa_modules.filter_utils import is_void_cluster
            if self.object_utils and self.object_utils.taxonomy_utils:
                clusters = self.object_utils.taxonomy_utils.get_object_clusters(matching_objects[0], 'final_taxonomy_physical_properties')
                if clusters and all(is_void_cluster(cluster, 'physical') for cluster in clusters):
                    return questions
            if not self.object_utils:
                return questions
            properties = self.object_utils.get_object_physical_properties(matching_objects[0])
            if not properties:
                return questions
            physical_property = properties[0]
        elif question_type == 'function_knowledge':
            from modules.qa_modules.filter_utils import is_void_cluster
            if self.object_utils and self.object_utils.taxonomy_utils:
                if self.object_utils.taxonomy_utils:
                    clusters = self.object_utils.taxonomy_utils.get_object_clusters(matching_objects[0], 'final_taxonomy_function')
                    if clusters and any(is_void_cluster(cluster, 'function') for cluster in clusters):
                        return questions
            # Check if function data exists
            if not self.object_utils:
                return questions
            function = self.object_utils.get_object_function(matching_objects[0])
            if not function or not function.strip():
                return questions
        elif question_type == 'description_matching':
            if not self.object_utils:
                return questions
            
            target_object = matching_objects[0]
            
            # Step 1: Filter by void clusters (affordance, shape, physical, texture)
            # Note: This requires access to unified_qa_generation_utils, but we can check void clusters directly
            from modules.qa_modules.filter_utils import is_void_cluster
            void_cluster_found = False
            if self.object_utils and self.object_utils.taxonomy_utils:
                taxonomy_types_to_check = ['affordance', 'shape', 'physical', 'texture']
                for tax_type in taxonomy_types_to_check:
                    try:
                        clusters = self.object_utils.taxonomy_utils.get_object_clusters(target_object, f'final_taxonomy_{tax_type}')
                        if clusters and any(is_void_cluster(cluster, tax_type) for cluster in clusters):
                            void_cluster_found = True
                            logger.debug(f"Skipping description_matching for {target_object}: object is in {tax_type} void cluster")
                            break
                    except Exception:
                        continue
            
            if void_cluster_found:
                return questions
            
            # Step 2: Check for description conflicts (if unified_qa_generation_utils is available)
            # For now, we'll do a simpler check here
            other_objects = [obj for obj in available_objects if obj != target_object]
            description = self.object_utils.get_object_description(target_object)
            if not description or not description.strip():
                return questions
            
            target_material = self.object_utils.get_object_material(target_object)
            target_function = self.object_utils.get_object_function(target_object)
            
            # Simple conflict check: if another object has same material/function
            conflicts_found = 0
            for other_obj in other_objects:
                other_material = self.object_utils.get_object_material(other_obj)
                other_function = self.object_utils.get_object_function(other_obj)
                
                # If same material AND same function → conflict
                if target_material and other_material and target_material.strip() and other_material.strip():
                    if target_material.strip().lower() == other_material.strip().lower():
                        if target_function and other_function and target_function.strip() and other_function.strip():
                            if target_function.strip().lower() == other_function.strip().lower():
                                conflicts_found += 1
                                break
            
            if conflicts_found > 0:
                logger.debug(f"Skipping description_matching for {target_object}: description conflicts with other objects")
                return questions
        
        try:
            question_text = get_question_template(question_type)
        except Exception as e:
            logger.warning(f"Could not get template for {question_type}: {e}")
            return questions
        
        # Format question text with available placeholders
        format_kwargs = {
            'objects': ', '.join(matching_objects),
            'object1': matching_objects[0] if len(matching_objects) > 0 else '',
            'object2': matching_objects[1] if len(matching_objects) > 1 else '',
            'object3': matching_objects[2] if len(matching_objects) > 2 else '',
        }

        if material and material.strip():
            format_kwargs['material'] = material
        if function and function.strip():
            format_kwargs['function'] = function
        if physical_property and physical_property.strip():
            format_kwargs['physical_property'] = physical_property
            format_kwargs.setdefault('property', physical_property)
        if description and description.strip():
            format_kwargs['description'] = description

        try:
            question_text = question_text.format(**format_kwargs)
        except KeyError as e:
            logger.error(f"Could not format question template for {question_type}: missing placeholder {e}")
            return questions
        
        # Determine answer (use first matching object)
        answer = matching_objects[0] if matching_objects else ''
        target_object = matching_objects[0] if matching_objects else None
        
        # Generate reasoning
        cot_generator = CoTReasoningGenerator(taxonomy_utils=self.taxonomy_utils)
        # Pass material/function/description to reasoning generator
        reasoning_kwargs = {}
        if material and material.strip():
            reasoning_kwargs['material'] = material
        if physical_property and physical_property.strip():
            reasoning_kwargs['physical_property'] = physical_property
        if function and function.strip():
            reasoning_kwargs['function'] = function
        if description and description.strip():
            reasoning_kwargs['description'] = description
        reasoning = cot_generator.generate_comprehensive_reasoning(
            question_type, answer, available_objects, answer, **reasoning_kwargs
        )
        
        # Validate question_text before creating question_data (prevent malformed questions)
        if not question_text or not question_text.strip():
            logger.warning(f"Empty or None question_text for {question_type}, skipping question")
            return questions
        
        # Validate answer before creating question_data
        if not answer or not answer.strip():
            logger.warning(f"Empty or None answer for {question_type}, skipping question")
            return questions
        
        question_data = {
            'question': question_text,
            'answer': answer,
            'question_type': question_type,
            'target_object': target_object,
            'objects': available_objects,
            'choices': available_objects,  # Explicitly set choices to all available objects
            'reasoning': reasoning,
            'image_path': f"{image_id}/bbox.jpg",
            'image_id': image_id
        }
        
        questions.append(question_data)
        return questions
    
    def _generate_spatial_questions(self, question_type: str, matching_objects: List[str],
                                   available_objects: List[str], question_data: Dict[str, Any],
                                   image_id: str, spatial_relationship: Dict[str, str] = None,
                                   reference_object: str = None) -> List[Dict[str, Any]]:
        """Generate spatial questions with proper directional answers"""
        questions = []
        
        if not spatial_relationship or not reference_object:
            logger.warning(f"No spatial relationship data for {question_type}")
            return questions
        
        try:
            # Get question template
            question_text = get_question_template(question_type)
        except Exception as e:
            logger.warning(f"Could not get template for {question_type}: {e}")
            return questions
        
        # Clean object names before formatting
        cleaned_objects = [self.clean_description_punctuation(obj) for obj in matching_objects]
        cleaned_reference = self.clean_description_punctuation(reference_object) if reference_object else ''
        
        # Format question with cleaned objects
        question_text = question_text.format(
            objects=', '.join(cleaned_objects),
            object1=cleaned_objects[0] if len(cleaned_objects) > 0 else '',
            object2=cleaned_objects[1] if len(cleaned_objects) > 1 else '',
            object3=cleaned_objects[2] if len(cleaned_objects) > 2 else '',
            reference_object=cleaned_reference
        )
        
        # Get directional answer from spatial relationship
        answer = self._get_spatial_answer(question_type, spatial_relationship)
        
        # Skip questions with low confidence (unknown answers indicate ambiguous spatial relationships)
        if answer == 'unknown':
            logger.debug(f"Skipping {question_type} - insufficient confidence to determine spatial relationship")
            return questions
        
        # Determine target object and reference object
        target_object = matching_objects[0] if matching_objects else None
        
        # Generate reasoning with spatial context
        cot_generator = CoTReasoningGenerator(taxonomy_utils=self.taxonomy_utils)
        spatial_context = {
            'spatial_relationship': spatial_relationship,
            'reference_object': reference_object
        }
        
        reasoning = cot_generator.generate_comprehensive_reasoning(
            question_type, answer, available_objects, answer, spatial_context=spatial_context
        )
        
        # Validate question_text before creating question_data (prevent malformed questions)
        if not question_text or not question_text.strip():
            logger.warning(f"Empty or None question_text for {question_type}, skipping question")
            return questions
        
        # Validate answer before creating question_data
        if not answer or not answer.strip():
            logger.warning(f"Empty or None answer for {question_type}, skipping question")
            return questions
        
        directional_choices = {
            'spatial_left_right': ['left', 'right'],
            'spatial_above_below': ['above', 'below'],
            'spatial_front_behind': ['front', 'behind']
        }

        question_data = {
            'question': question_text,
            'answer': answer,
            'question_type': question_type,
            'target_object': target_object,
            'objects': available_objects,
            'choices': directional_choices.get(question_type, available_objects),
            'reasoning': reasoning,
            'image_path': f"{image_id}/bbox.jpg",
            'image_id': image_id,
            'spatial_context': spatial_context
        }
        
        questions.append(question_data)
        return questions
    
    def _get_spatial_answer(self, question_type: str, spatial_relationship: Dict[str, str]) -> str:
        """Get the correct directional answer for spatial questions"""
        if question_type == 'spatial_left_right':
            return spatial_relationship.get('left_right', 'unknown')
        elif question_type == 'spatial_above_below':
            return spatial_relationship.get('above_below', 'unknown')
        elif question_type == 'spatial_front_behind':
            return spatial_relationship.get('front_behind', 'unknown')
        elif question_type == 'spatial_closer_to_camera':
            return spatial_relationship.get('closer', 'unknown')
        else:
            return 'unknown'
    
    def generate_spatial_questions_sim(self, question_type: str, objects: List[str], 
                                     scene_id: str, cot_generator: CoTReasoningGenerator,
                                     object_poses: Dict[str, Dict[str, Any]], scene_path: Path = None) -> List[Dict[str, Any]]:
        """Generate spatial questions for sim images using pose data"""
        questions = []
        
        if len(objects) < 2:
            return questions
        
        # Get spatial relationship from pose data
        spatial_info = self._get_spatial_relationship_from_poses(objects, object_poses, question_type, scene_path)
        
        if not spatial_info:
            return questions
        
        # Skip questions with low confidence (unknown or ambiguous spatial relationships)
        if spatial_info.get('answer') == 'unknown' or not spatial_info.get('answer'):
            logger.debug(f"Skipping {question_type} - insufficient confidence to determine spatial relationship")
            return questions
        
        try:
            # Get question template with spatial parameters and fill in object names
            # Only support the active question types
            if question_type in ['spatial_front_behind', 'spatial_above_below', 'spatial_left_right', 'spatial_closer_to_camera']:
                question_text = get_question_template(question_type, object1=objects[0], object2=objects[1])
            else:
                # Skip unsupported question types
                return questions
        except Exception as e:
            logger.warning(f"Could not get template for {question_type}: {e}")
            return questions
        
        # Generate reasoning with spatial context
        spatial_context = {
            'spatial_relationship': spatial_info.get('relationship'),
            'object_positions': {obj: object_poses.get(obj, {}) for obj in objects},
            'calculation_details': spatial_info.get('calculation_details')
        }
        
        reasoning = cot_generator.generate_comprehensive_reasoning(
            question_type, spatial_info['answer'], objects, spatial_info['answer'], 
            spatial_context=spatial_context
        )
        
        # Validate question_text before creating question_data (prevent malformed questions)
        if not question_text or not question_text.strip():
            logger.warning(f"Empty or None question_text for {question_type}, skipping question")
            return questions
        
        # Validate answer before creating question_data
        answer = spatial_info['answer']
        if not answer or not answer.strip():
            logger.warning(f"Empty or None answer for {question_type}, skipping question")
            return questions
        
        directional_choices = {
            'spatial_left_right': ['left', 'right'],
            'spatial_above_below': ['above', 'below'],
            'spatial_front_behind': ['front', 'behind']
        }

        question_data = {
            'question': question_text,
            'answer': answer,
            'question_type': question_type,
            'target_object': objects[0] if objects else None,
            'objects': objects,
            'choices': directional_choices.get(question_type, objects),
            'reasoning': reasoning,
            'image_path': f"{scene_id}/bbox.jpg",
            'image_id': scene_id,  # Set image_id for consistency
            'scene_id': scene_id
        }
        
        questions.append(question_data)
        return questions
    
    def _get_spatial_relationship_from_poses(self, objects: List[str], 
                                           object_poses: Dict[str, Dict[str, Any]], 
                                           question_type: str, scene_path: Path = None) -> Dict[str, Any]:
        """Determine spatial relationship from 3D pose data for sim images (with optional depth support)."""
        if len(objects) < 2:
            return None

        def _to_vector(data: Any) -> Optional[np.ndarray]:
            if isinstance(data, (list, tuple)) and len(data) == 3:
                try:
                    return np.array([float(x) for x in data], dtype=float)
                except Exception:
                    return None
            return None

        def _compute_yaw(front_vec: Optional[np.ndarray]) -> Optional[float]:
            if front_vec is None:
                return None
            try:
                yaw = math.degrees(math.atan2(front_vec[0], front_vec[2]))
                if yaw > 180.0:
                    yaw -= 360.0
                if yaw < -180.0:
                    yaw += 360.0
                return yaw
            except Exception:
                return None

        def _build_object_details(obj_name: str, pixel_center: Optional[Tuple[float, float]]) -> Dict[str, Any]:
            pose = object_poses.get(obj_name, {})
            front_vec = _to_vector(pose.get('front'))
            return {
                'object': obj_name,
                'center': pixel_center,
                'yaw_deg': _compute_yaw(front_vec),
            }

        # Optional depth-based shortcut (rarely used); fall back to detailed logic if unavailable
        if self.object_utils and scene_path:
            depth_map_path = scene_path / "depth.npy"
            if depth_map_path.exists():
                spatial_info = self.object_utils.get_spatial_relationship(
                    objects[0], objects[1], "dummy_scene_id", object_poses=object_poses, depth_map_path=depth_map_path
                )
                if spatial_info:
                    mapping = {
                        'spatial_front_behind': spatial_info.get('front_behind', 'unknown'),
                        'spatial_above_below': spatial_info.get('above_below', 'unknown'),
                        'spatial_left_right': spatial_info.get('left_right', 'unknown'),
                    }
                    if question_type in mapping:
                        answer = mapping[question_type]
                        if answer != 'unknown':
                            return {
                                'answer': answer,
                                'object1': objects[0],
                                'object2': objects[1],
                                'relationship': question_type.replace('spatial_', ''),
                                'calculation_details': None,
                            }

        # Extract world positions and bbox centers
        positions = {}
        bbox_centers: Dict[str, Optional[Tuple[float, float]]] = {}
        for obj in objects:
            pose = object_poses.get(obj, {})
            positions[obj] = pose.get('location', [0, 0, 0])
            bbox = pose.get('bbox_2d')
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
                    cy = (float(bbox[1]) + float(bbox[3])) / 2.0
                    bbox_centers[obj] = (cx, cy)
                except Exception:
                    bbox_centers[obj] = None
            else:
                bbox_centers[obj] = None

        obj1, obj2 = objects[0], objects[1]
        pos1 = np.array(positions[obj1], dtype=float)
        pos2 = np.array(positions[obj2], dtype=float)
        relative_vec = pos1 - pos2
        center1 = bbox_centers.get(obj1)
        center2 = bbox_centers.get(obj2)
        min_separation = 0.05  # 5 cm threshold

        def _build_details(relation_value: str, description: str) -> Dict[str, Any]:
            return {
                'reference': _build_object_details(obj2, center2),
                'target': _build_object_details(obj1, center1),
                'delta': {
                    'dx': float(pos1[0] - pos2[0]),
                    'dy': float(pos1[1] - pos2[1]),
                    'dz': float(pos1[2] - pos2[2]),
                    'axis_description': description,
                },
                'relation_value': relation_value,
            }

        if question_type == 'spatial_front_behind':
            front_vec = _to_vector(object_poses.get(obj2, {}).get('front'))
            class2 = object_poses.get(obj2, {}).get('class_name')
            if (
                self.object_utils
                and front_vec is not None
                and self.object_utils._is_high_reliability_orientation(class2)
            ):
                orientation_result = self.object_utils._calculate_front_behind_from_orientation(
                    {
                        'pcd_center': pos2.tolist(),
                        'front': object_poses[obj2].get('front'),
                        'left': object_poses[obj2].get('left'),
                    },
                    {'pcd_center': pos1.tolist()},
                )
                if orientation_result in ('front', 'behind'):
                    signed = float(np.dot(relative_vec, front_vec))
                    direction = 'front' if signed > 0 else 'behind'
                    axis_desc = (
                        f"Projection onto {obj2}'s front axis is {signed:.2f} units toward the {direction}."
                    )
                    return {
                        'answer': orientation_result,
                        'object1': obj1,
                        'object2': obj2,
                        'relationship': 'front_behind',
                        'calculation_details': _build_details(orientation_result, axis_desc),
                    }
            depth_sep = abs(float(pos1[1] - pos2[1]))
            if depth_sep < min_separation:
                return {
                    'answer': 'unknown',
                    'object1': obj1,
                    'object2': obj2,
                    'relationship': 'front_behind',
                    'calculation_details': _build_details('unknown', "Depth separation is below threshold."),
                }
            relation = 'front' if pos1[1] < pos2[1] else 'behind'
            axis_desc = f"World depth offset is {pos1[1] - pos2[1]:.2f} units, so {obj1} is {relation} of {obj2}."
            return {
                'answer': relation,
                'object1': obj1,
                'object2': obj2,
                'relationship': 'front_behind',
                'calculation_details': _build_details(relation, axis_desc),
            }

        if question_type == 'spatial_above_below':
            z_diff = abs(float(pos1[2] - pos2[2]))
            ground_level_threshold = 0.15
            both_on_ground = pos1[2] < ground_level_threshold and pos2[2] < ground_level_threshold
            if both_on_ground:
                return {
                    'answer': 'unknown',
                    'object1': obj1,
                    'object2': obj2,
                    'relationship': 'above_below',
                    'calculation_details': _build_details('unknown', "Both objects rest near ground level."),
                }
            if z_diff < min_separation:
                return {
                    'answer': 'unknown',
                    'object1': obj1,
                    'object2': obj2,
                    'relationship': 'above_below',
                    'calculation_details': _build_details('unknown', "Vertical separation is below threshold."),
                }
            relation = 'above' if pos1[2] > pos2[2] else 'below'
            axis_desc = f"World height offset is {pos1[2] - pos2[2]:.2f} units, so {obj1} is {relation} {obj2}."
            return {
                'answer': relation,
                'object1': obj1,
                'object2': obj2,
                'relationship': 'above_below',
                'calculation_details': _build_details(relation, axis_desc),
            }

        if question_type == 'spatial_left_right':
            left_vec = _to_vector(object_poses.get(obj2, {}).get('left'))
            front_vec = _to_vector(object_poses.get(obj2, {}).get('front'))
            class2 = object_poses.get(obj2, {}).get('class_name')
            if (
                self.object_utils
                and left_vec is not None
                and front_vec is not None
                and self.object_utils._is_high_reliability_orientation(class2)
            ):
                orientation_result = self.object_utils._calculate_left_right_from_orientation(
                    {
                        'pcd_center': pos2.tolist(),
                        'left': object_poses[obj2].get('left'),
                        'front': object_poses[obj2].get('front'),
                    },
                    {'pcd_center': pos1.tolist()},
                )
                if orientation_result in ('left', 'right'):
                    signed = float(np.dot(relative_vec, left_vec))
                    direction = 'left' if signed > 0 else 'right'
                    axis_desc = (
                        f"Projection onto {obj2}'s left axis is {signed:.2f} units toward the {direction}."
                    )
                    return {
                        'answer': orientation_result,
                        'object1': obj1,
                        'object2': obj2,
                        'relationship': 'left_right',
                        'calculation_details': _build_details(orientation_result, axis_desc),
                    }
            if center1 and center2:
                x_sep_px = abs(center1[0] - center2[0])
                if x_sep_px >= 10.0:
                    relation = 'right' if center1[0] > center2[0] else 'left'
                    axis_desc = (
                        f"Horizontal bbox center offset is {center1[0] - center2[0]:.1f}px, so {obj1} is to the {relation} of {obj2}."
                    )
                    return {
                        'answer': relation,
                        'object1': obj1,
                        'object2': obj2,
                        'relationship': 'left_right',
                        'calculation_details': _build_details(relation, axis_desc),
                    }
            x_diff = abs(float(pos1[0] - pos2[0]))
            if x_diff < min_separation:
                return {
                    'answer': 'unknown',
                    'object1': obj1,
                    'object2': obj2,
                    'relationship': 'left_right',
                    'calculation_details': _build_details('unknown', "Horizontal separation is below threshold."),
                }
            relation = 'right' if pos1[0] > pos2[0] else 'left'
            axis_desc = f"World X offset is {pos1[0] - pos2[0]:.2f} units, so {obj1} is to the {relation} of {obj2}."
            return {
                'answer': relation,
                'object1': obj1,
                'object2': obj2,
                'relationship': 'left_right',
                'calculation_details': _build_details(relation, axis_desc),
            }

        if question_type == 'spatial_closer_to_camera':
            relation_obj = obj1 if pos1[1] < pos2[1] else obj2
            axis_desc = (
                f"World depth offset is {pos1[1] - pos2[1]:.2f} units; the smaller value is closer to camera."
            )
            return {
                'answer': relation_obj,
                'object1': obj1,
                'object2': obj2,
                'relationship': 'closer',
                'calculation_details': _build_details(relation_obj, axis_desc),
            }

        return {
            'answer': obj1,
            'object1': obj1,
            'object2': obj2,
            'relationship': 'unknown',
            'calculation_details': _build_details(obj1, "Default fallback used."),
        }
