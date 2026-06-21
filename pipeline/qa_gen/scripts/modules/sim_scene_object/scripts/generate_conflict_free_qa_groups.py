#!/usr/bin/env python3

import json
import sys
import random
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

# Add path to access QA modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from modules.qa_modules.question_templates import get_question_template
from collections import defaultdict

# Add the modules directory to the path
sys.path.append('/path/to/SpatialReasonerDataGen/qa_gen/scripts/modules')

# Import QA modules for proper question generation
# Note: We'll use a simpler approach without complex initialization

class ConflictFreeQAGroupGenerator:
    def __init__(self):
        self.question_answer_space = self._load_qa_space()
        self.placable_objects = self._load_placable_objects()
        self.object_mapping = self._load_object_mapping()
        self.scene_data = self._load_scene_data()
        self.qa_potential_analysis = self._load_qa_potential_analysis()
    
    def _load_qa_space(self) -> Dict[str, Any]:
        """Load the question-answer space analysis"""
        # Use path relative to qa_gen directory
        script_dir = Path(__file__).parent.parent.parent.parent.parent
        qa_space_file = script_dir / "scripts/analysis/results/question_answer_space_analysis.json"
        if qa_space_file.exists():
            with open(qa_space_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('question_answer_mappings', {})
        return {}
    
    def _load_placable_objects(self) -> Set[str]:
        """Load placable objects list"""
        placable_objects = set()
        
        # Load floor placable objects
        floor_file = Path("../data/mapping_placable/placable_on_floor_list.json")
        if floor_file.exists():
            try:
                with open(floor_file, 'r', encoding='utf-8') as f:
                    floor_data = json.load(f)
                    for category, category_data in floor_data.items():
                        if isinstance(category_data, dict) and 'meshes' in category_data:
                            for mesh in category_data['meshes']:
                                if isinstance(mesh, dict) and 'object name' in mesh:
                                    placable_objects.add(mesh['object name'])
                print(f"Loaded {len(placable_objects)} floor placable objects")
            except Exception as e:
                print(f"Warning: Could not load floor placable objects: {e}")
        
        return placable_objects
    
    def _load_object_mapping(self) -> Dict[str, str]:
        """Load SM to human object mapping"""
        mapping_file = Path("../data/all_objects_with_scenes.json")
        sm_to_human = {}
        
        if mapping_file.exists():
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                
                for human_name, objects in mapping_data.items():
                    for obj in objects:
                        if isinstance(obj, dict) and 'object name' in obj:
                            sm_name = obj['object name']
                            sm_to_human[sm_name] = human_name
                
                print(f"Loaded {len(sm_to_human)} object mappings")
            except Exception as e:
                print(f"Warning: Could not load object mapping: {e}")
        
        return sm_to_human
    
    def _load_scene_data(self) -> Dict[str, List[str]]:
        """Load scene data with objects"""
        scene_data = {}
        
        # Load from actual scene files with objects
        scene_dir = Path("../data/scene_with_object")
        if scene_dir.exists():
            for scene_file in scene_dir.glob("*_object_category.json"):
                scene_name = scene_file.stem
                try:
                    with open(scene_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract all SM object names from the scene
                    objects = []
                    for category, sm_objects in data.items():
                        if isinstance(sm_objects, list):
                            objects.extend(sm_objects)
                        elif isinstance(sm_objects, str):
                            objects.append(sm_objects)
                    
                    scene_data[scene_name] = objects
                except Exception as e:
                    print(f"Warning: Could not load scene {scene_name}: {e}")
        
        return scene_data
    
    def _load_qa_potential_analysis(self) -> Dict[str, Any]:
        """Load the QA potential analysis"""
        analysis_file = Path("scene_qa_potential_analysis.json")
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _filter_placable_objects(self, objects: List[str]) -> List[str]:
        """Filter objects to only include placable ones"""
        return [obj for obj in objects if obj in self.placable_objects]
    
    def _get_unique_human_categories(self, placable_objects: List[str]) -> Dict[str, str]:
        """Get unique human categories from placable objects, mapping to one representative SM object"""
        human_to_sm = {}
        for sm_object in placable_objects:
            human_name = self.object_mapping.get(sm_object)
            if human_name and human_name not in human_to_sm:
                human_to_sm[human_name] = sm_object
        return human_to_sm
    
    def _get_question_types_for_human(self, human_name: str) -> Set[str]:
        """Get question types that a human category can answer"""
        question_types = set()
        for question_type, qa_data in self.question_answer_space.items():
            if qa_data.get('sm_valid_objects') and human_name in qa_data['sm_valid_objects']:
                question_types.add(question_type)
        return question_types
    
    def _find_conflict_free_group_by_category(self, remaining_humans: List[str], human_to_sm: Dict[str, str], group_size: int = 6) -> Tuple[List[str], List[str]]:
        """Find a conflict-free group of human categories"""
        if len(remaining_humans) < group_size:
            return remaining_humans, []
        
        # Start with the first human category
        first_human = remaining_humans[0]
        first_question_types = self._get_question_types_for_human(first_human)
        
        # Find human categories that don't conflict with the first one
        compatible_humans = []
        conflicting_humans = []
        
        for human in remaining_humans[1:]:
            human_question_types = self._get_question_types_for_human(human)
            
            # Check for conflicts (overlapping question types)
            overlap = first_question_types.intersection(human_question_types)
            conflict_ratio = len(overlap) / len(first_question_types) if first_question_types else 0
            
            if conflict_ratio < 0.3:  # Less than 30% overlap = compatible
                compatible_humans.append(human)
            else:
                conflicting_humans.append(human)
        
        # Select up to group_size-1 compatible human categories
        if len(compatible_humans) >= group_size - 1:
            selected_compatible = random.sample(compatible_humans, group_size - 1)
        else:
            selected_compatible = compatible_humans
            # If not enough compatible categories, add some conflicting ones
            remaining_needed = group_size - 1 - len(selected_compatible)
            if remaining_needed > 0 and conflicting_humans:
                additional = random.sample(conflicting_humans, min(remaining_needed, len(conflicting_humans)))
                selected_compatible.extend(additional)
        
        # Create the group of human categories
        group_humans = [first_human] + selected_compatible
        
        # Convert back to SM objects
        group_objects = [human_to_sm[human] for human in group_humans]
        
        # Remove used human categories from remaining
        used_humans = set(group_humans)
        new_remaining = [human for human in remaining_humans if human not in used_humans]
        
        return group_objects, new_remaining
    
    def _generate_example_qa_for_group(self, group_objects: List[str], group_human_categories: List[str], scene_name: str) -> Dict[str, Any]:
        """Generate one example QA for the group using proper 'Which object...' format"""
        if not group_objects or not group_human_categories:
            return None

        # Select a random target object from the group
        target_sm_object = random.choice(group_objects)
        target_human_name = self.object_mapping.get(target_sm_object)
        if not target_human_name:
            return None

        # Find question types this target object can answer
        available_question_types = []
        for question_type, qa_data in self.question_answer_space.items():
            if qa_data.get('sm_valid_objects') and target_human_name in qa_data['sm_valid_objects']:
                available_question_types.append(question_type)

        if not available_question_types:
            return None

        # Select a random question type
        selected_question_type = random.choice(available_question_types)
        qa_data = self.question_answer_space[selected_question_type]
        
        # Use the centralized question templates from the modules
        question = get_question_template(selected_question_type)

        return {
            'question_type': selected_question_type,
            'question': question,
            'answer': target_human_name,  # The human category name as answer
            'target_object': target_sm_object,  # The SM object as target
            'options': group_human_categories,  # All human categories as options
            'reasoning': f"Looking at the objects in the scene, I can identify which one matches the specified properties.",
            'scene_name': scene_name
        }
    
    
    def _process_scene(self, scene_name: str, objects: List[str]) -> Dict[str, Any]:
        """Process a single scene to create conflict-free groups"""
        placable_objects = self._filter_placable_objects(objects)
        
        if len(placable_objects) < 2:
            return {
                "scene_name": scene_name,
                "total_objects": len(objects),
                "placable_objects": len(placable_objects),
                "groups": [],
                "total_qa_pairs": 0
            }
        
        # Get unique human categories from placable objects
        human_to_sm = self._get_unique_human_categories(placable_objects)
        unique_humans = list(human_to_sm.keys())
        
        print(f"Processing scene: {scene_name}")
        print(f"  Starting with {len(placable_objects)} placable objects -> {len(unique_humans)} unique categories")
        
        if len(unique_humans) < 2:
            return {
                "scene_name": scene_name,
                "total_objects": len(objects),
                "placable_objects": len(placable_objects),
                "unique_categories": len(unique_humans),
                "groups": [],
                "total_qa_pairs": 0
            }
        
        groups = []
        remaining_humans = unique_humans.copy()
        random.shuffle(remaining_humans)  # Shuffle for diversity
        total_qa_pairs = 0
        
        group_index = 0
        while len(remaining_humans) >= 6:
            # Create a conflict-free group by human categories
            group_objects, remaining_humans = self._find_conflict_free_group_by_category(remaining_humans, human_to_sm)
            
            # Generate one example QA for the group
            group_human_categories = [self.object_mapping.get(obj) for obj in group_objects]
            example_qa = self._generate_example_qa_for_group(group_objects, group_human_categories, scene_name)
            
            if example_qa:
                total_qa_pairs += 1
            
            groups.append({
                "group_index": group_index,
                "objects": group_objects,
                "human_categories": [self.object_mapping.get(obj) for obj in group_objects],
                "example_qa": example_qa
            })
            
            group_index += 1
            print(f"  Group {group_index}: {len(group_objects)} objects, QA: {'Yes' if example_qa else 'No'}")
        
        # Handle remaining human categories
        if remaining_humans:
            # Create a final group with remaining categories
            remaining_objects = [human_to_sm[human] for human in remaining_humans]
            remaining_human_categories = [self.object_mapping.get(obj) for obj in remaining_objects]
            example_qa = self._generate_example_qa_for_group(remaining_objects, remaining_human_categories, scene_name)
            
            if example_qa:
                total_qa_pairs += 1
            
            groups.append({
                "group_index": group_index,
                "objects": remaining_objects,
                "human_categories": [self.object_mapping.get(obj) for obj in remaining_objects],
                "example_qa": example_qa
            })
            
            print(f"  Final group: {len(remaining_objects)} objects, QA: {'Yes' if example_qa else 'No'}")
        
        print(f"  Total groups: {len(groups)}, Total QA pairs: {total_qa_pairs}")
        
        return {
            "scene_name": scene_name,
            "total_objects": len(objects),
            "placable_objects": len(placable_objects),
            "unique_categories": len(unique_humans),
            "groups": groups,
            "total_qa_pairs": total_qa_pairs
        }
    
    def generate_all_scenes(self) -> Dict[str, Any]:
        """Generate conflict-free groups for all scenes"""
        results = {}
        total_groups = 0
        total_qa_pairs = 0
        scenes_processed = 0
        
        print(f"Generating conflict-free QA groups for {len(self.scene_data)} scenes...")
        
        for scene_name, objects in self.scene_data.items():
            scene_result = self._process_scene(scene_name, objects)
            results[scene_name] = scene_result
            
            total_groups += len(scene_result['groups'])
            total_qa_pairs += scene_result['total_qa_pairs']
            scenes_processed += 1
        
        return {
            "summary": {
                "total_scenes": len(self.scene_data),
                "scenes_processed": scenes_processed,
                "total_groups": total_groups,
                "total_qa_pairs": total_qa_pairs,
                "average_groups_per_scene": total_groups / scenes_processed if scenes_processed > 0 else 0,
                "average_qa_per_scene": total_qa_pairs / scenes_processed if scenes_processed > 0 else 0
            },
            "scene_results": results
        }
    
    def save_results(self, output_file: Path):
        """Save the results to file"""
        results = self.generate_all_scenes()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Total scenes: {results['summary']['total_scenes']}")
        print(f"Total groups: {results['summary']['total_groups']}")
        print(f"Total QA pairs: {results['summary']['total_qa_pairs']}")
        print(f"Average groups per scene: {results['summary']['average_groups_per_scene']:.1f}")
        print(f"Average QA per scene: {results['summary']['average_qa_per_scene']:.1f}")

def main():
    generator = ConflictFreeQAGroupGenerator()
    output_file = Path("conflict_free_qa_groups.json")
    generator.save_results(output_file)

if __name__ == "__main__":
    main()