#!/usr/bin/env python3
"""
Taxonomy QA Generator Base Classes and Utilities

Common functionality for taxonomy-based question generation.
"""

import random
import logging
from typing import Dict, List, Any, Tuple
from .taxonomy_utils import TaxonomyUtils

logger = logging.getLogger(__name__)


class TaxonomyQABase:
    """Base class for taxonomy-based QA generation"""
    
    def __init__(self, taxonomy_utils: TaxonomyUtils, object_descriptions: Dict[str, Dict[str, Any]]):
        self.taxonomy_utils = taxonomy_utils
        self.object_descriptions = object_descriptions
    
    def _extract_object_name(self, obj: Dict) -> str:
        """Extract object name from object dict"""
        return obj.get('labeled_name', obj['class_name'])
    
    def _format_question_with_scene_objects(self, question_text: str, scene_objects: List[str]) -> str:
        """Format question to ask for object name without limiting options to specific objects"""
        # Check if this is a question that asks for an object name
        object_question_patterns = [
            "which object",
            "which of the following",
            "what object",
            "name the object",
            "identify the object"
        ]
        
        question_lower = question_text.lower()
        is_object_question = any(pattern in question_lower for pattern in object_question_patterns)
        
        if is_object_question:
            # Remove trailing question mark if present
            if question_text.endswith("?"):
                question_text = question_text[:-1]
            return f"{question_text}? Please provide the name of the object you think answers this question. All bounded objects are possible choices."
        
        return question_text
    
    def _get_taxonomic_reasoning_for_object(self, obj_name: str) -> Dict:
        """Get taxonomic reasoning for any object"""
        try:
            material_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            physical_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_physical_properties')
            affordance_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_affordances')
            texture_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_texture')
            shape_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_shape')
            function_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_function')
        except Exception as e:
            return {"error": f"Could not retrieve taxonomy for {obj_name}: {e}"}
        
        return {
            "object_name": obj_name,
            "taxonomy_clusters": {
                "material": material_clusters,
                "physical_properties": physical_clusters,
                "affordances": affordance_clusters,
                "texture": texture_clusters,
                "shape": shape_clusters,
                "function": function_clusters
            }
        }
    
    def _create_chain_structured_reasoning(self, question_type: str, answer: str, 
                                         detected_objects: List[Dict], 
                                         filter_chain: List[Dict]) -> Dict:
        """
        Create chain-structured reasoning similar to the diagram shown.
        Each step in the chain represents a filter that narrows down to the correct answer.
        Shows both objects that pass and objects that fail each filter.
        
        Args:
            question_type: Type of question (e.g., 'material_description', 'repurposing_doorstop')
            answer: The correct answer object
            detected_objects: All objects in the scene
            filter_chain: List of filter steps with format:
                [
                    {"filter_type": "Filter material", "filter_value": "steel", "objects_after": ["rifle", "guard"], "objects_before": ["rifle", "guard", "chair"]},
                    {"filter_type": "Filter shape", "filter_value": "elongated", "objects_after": ["rifle"], "objects_before": ["rifle", "guard"]},
                    {"filter_type": "Unique", "filter_value": None, "objects_after": ["rifle"], "objects_before": ["rifle"]}
                ]
        """
        reasoning = {
            "question_type": question_type,
            "answer": answer,
            "total_objects": len(detected_objects),
            "filter_chain": filter_chain,
            "chain_summary": self._generate_chain_summary(filter_chain, answer),
            "reasoning_flow": self._generate_reasoning_flow(filter_chain),
            "elimination_process": self._generate_elimination_process(filter_chain)
        }
        
        return reasoning
    
    def _generate_chain_summary(self, filter_chain: List[Dict], answer: str) -> str:
        """Generate a natural language summary of the filter chain"""
        if not filter_chain:
            return f"Selected {answer} as the answer."
        
        steps = []
        for i, step in enumerate(filter_chain):
            filter_type = step.get("filter_type", "")
            filter_value = step.get("filter_value", "")
            objects_after = step.get("objects_after", [])
            
            if filter_type == "Unique":
                steps.append(f"Step {i+1}: {filter_type} → {objects_after[0] if objects_after else 'No objects'}")
            elif filter_value:
                steps.append(f"Step {i+1}: {filter_type} '{filter_value}' → {len(objects_after)} objects: {objects_after}")
            else:
                steps.append(f"Step {i+1}: {filter_type} → {len(objects_after)} objects: {objects_after}")
        
        return " → ".join([f"Step {i+1}" for i in range(len(steps))]) + f" → Answer: {answer}"
    
    def _generate_reasoning_flow(self, filter_chain: List[Dict]) -> List[str]:
        """Generate detailed reasoning flow for each step"""
        flow = []
        for i, step in enumerate(filter_chain):
            filter_type = step.get("filter_type", "")
            filter_value = step.get("filter_value", "")
            objects_after = step.get("objects_after", [])
            
            if filter_type == "Unique":
                flow.append(f"Step {i+1} (Unique): Identified unique object: {objects_after[0] if objects_after else 'None'}")
            elif filter_value:
                flow.append(f"Step {i+1} ({filter_type}): Applied filter '{filter_value}' → Found {len(objects_after)} matching objects: {objects_after}")
            else:
                flow.append(f"Step {i+1} ({filter_type}): Applied operation → Result: {objects_after}")
        
        return flow
    
    def _generate_elimination_process(self, filter_chain: List[Dict]) -> List[Dict]:
        """Generate detailed elimination process showing which objects fail each filter"""
        elimination_steps = []
        
        for i, step in enumerate(filter_chain):
            filter_type = step.get("filter_type", "")
            filter_value = step.get("filter_value", "")
            objects_before = step.get("objects_before", [])
            objects_after = step.get("objects_after", [])
            
            # Find objects that were eliminated
            eliminated_objects = [obj for obj in objects_before if obj not in objects_after]
            
            elimination_step = {
                "step_number": i + 1,
                "filter_type": filter_type,
                "filter_value": filter_value,
                "objects_before": objects_before,
                "objects_after": objects_after,
                "objects_eliminated": eliminated_objects,
                "elimination_reason": self._get_elimination_reason(filter_type, filter_value, eliminated_objects),
                "survivors": objects_after
            }
            
            elimination_steps.append(elimination_step)
        
        return elimination_steps
    
    def _get_elimination_reason(self, filter_type: str, filter_value: str, eliminated_objects: List[str]) -> str:
        """Generate explanation for why objects were eliminated"""
        if not eliminated_objects:
            return "No objects eliminated"
        
        if filter_type == "Unique":
            return f"All objects except the final answer were already eliminated in previous steps"
        
        # Create specific elimination reasons based on filter type
        if "material" in filter_type.lower():
            return f"Objects {eliminated_objects} eliminated because they are not made of '{filter_value}'"
        elif "shape" in filter_type.lower():
            return f"Objects {eliminated_objects} eliminated because they don't have '{filter_value}' shape"
        elif "physical" in filter_type.lower():
            return f"Objects {eliminated_objects} eliminated because they don't have '{filter_value}' physical property"
        elif "thermal" in filter_type.lower():
            return f"Objects {eliminated_objects} eliminated because they don't have '{filter_value}' thermal property"
        elif "repurposing" in filter_type.lower():
            return f"Objects {eliminated_objects} eliminated because they don't match '{filter_value}' repurposing concept"
        elif "required" in filter_type.lower():
            return f"Objects {eliminated_objects} eliminated because they don't have required properties: '{filter_value}'"
        elif "forbidden" in filter_type.lower():
            return f"Objects {eliminated_objects} eliminated because they have forbidden properties: '{filter_value}'"
        else:
            return f"Objects {eliminated_objects} eliminated because they don't match filter '{filter_value}'"
    

    def _create_question(self, question_text: str, answer: str, question_type: str, 
                        difficulty: str, objects_involved: List[str], 
                        scene_object_names: List[str], 
                        reasoning_categories: List[str] = None,
                        metadata: Dict = None,
                        filter_chain: List[Dict] = None) -> Dict:
        """Create a standardized question dictionary with chain-structured reasoning"""
        # Create chain-structured reasoning if filter_chain is provided
        if filter_chain:
            chain_reasoning = self._create_chain_structured_reasoning(
                question_type, answer, objects_involved, filter_chain
            )
        else:
            # Fallback to basic taxonomic reasoning for spatial questions
            chain_reasoning = self._get_taxonomic_reasoning_for_object(answer)
        
        return {
            "question": self._format_question_with_scene_objects(question_text, scene_object_names),
            "answer": answer,
            "question_type": question_type,
            "difficulty": difficulty,
            "objects_involved": objects_involved,
            "reasoning_categories": reasoning_categories or [],
            "metadata": metadata or {},
            "chain_reasoning": chain_reasoning
        }
    
    def _filter_by_taxonomy(self, detected_objects: List[Dict], taxonomy_name: str, 
                           cluster_names: List[str]) -> Dict[str, List[Dict]]:
        """Filter objects by taxonomy clusters"""
        result = {}
        for cluster_name in cluster_names:
            result[cluster_name] = []
            for obj in detected_objects:
                obj_name = self._extract_object_name(obj)
                class_name = obj.get('class_name', '').lower()
                
                # Exclude human objects from material-related taxonomy filtering
                human_exclusions = ['person', 'man', 'woman', 'child', 'baby', 'guard', 'soldier', 'army', 'police', 'officer']
                if any(exclusion in class_name for exclusion in human_exclusions):
                    continue
                if any(exclusion in obj_name.lower() for exclusion in human_exclusions):
                    continue
                
                clusters = self.taxonomy_utils.get_object_clusters(obj_name, taxonomy_name)
                if cluster_name in clusters:
                    result[cluster_name].append(obj)
        return result
    
    def _filter_by_physical_property(self, detected_objects: List[Dict], properties: List[str]) -> List[Dict]:
        """Filter objects by physical properties"""
        result = []
        for obj in detected_objects:
            obj_name = self._extract_object_name(obj)
            obj_properties = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_physical_properties')
            if any(prop in obj_properties for prop in properties):
                result.append(obj)
        return result
    
    def _filter_by_shape(self, detected_objects: List[Dict], shapes: List[str]) -> List[Dict]:
        """Filter objects by shape characteristics"""
        result = []
        for obj in detected_objects:
            obj_name = self._extract_object_name(obj)
            obj_shapes = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_shape')
            if any(shape in obj_shapes for shape in shapes):
                result.append(obj)
        return result
    
    def _get_random_objects_with_choices(self, target_objects: List[Dict], 
                                       all_objects: List[Dict], 
                                       scene_object_names: List[str],
                                       num_choices: int = 3) -> Tuple[Dict, List[str]]:
        """Get a random target object and create choices list"""
        if not target_objects:
            return None, []
        
        target_obj = random.choice(target_objects)
        target_obj_name = self._extract_object_name(target_obj)
        
        # Create choices list
        all_obj_names = [self._extract_object_name(obj) for obj in all_objects]
        all_choices = [target_obj_name] + [name for name in all_obj_names if name != target_obj_name]
        random.shuffle(all_choices)
        
        return target_obj, all_choices[:num_choices]
