#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict

# Add the modules directory to the path
sys.path.append('/path/to/SpatialReasonerDataGen/qa_gen/scripts/modules')

class SceneQAPotentialAnalyzer:
    def __init__(self):
        self.question_answer_space = self._load_qa_space()
        self.placable_objects = self._load_placable_objects()
        self.object_mapping = self._load_object_mapping()
        self.scene_data = self._load_scene_data()
    
    def _load_qa_space(self) -> Dict[str, Any]:
        """Load the question-answer space analysis"""
        # Use path relative to qa_gen directory
        script_dir = Path(__file__).parent.parent.parent.parent.parent
        qa_space_file = script_dir / "scripts/analysis/results/question_answer_space_analysis.json"
        if qa_space_file.exists():
            with open(qa_space_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract question_answer_mappings from the data
                return data.get('question_answer_mappings', {})
        return {}
    
    def _load_placable_objects(self) -> Set[str]:
        """Load placable objects list"""
        placable_objects = set()
        
        # Load floor placable objects (same as working script)
        floor_file = Path("../data/mapping_placable/placable_on_floor_list.json")
        if floor_file.exists():
            try:
                with open(floor_file, 'r', encoding='utf-8') as f:
                    floor_data = json.load(f)
                    # Extract object names from nested structure
                    for category, category_data in floor_data.items():
                        if isinstance(category_data, dict) and 'meshes' in category_data:
                            for mesh in category_data['meshes']:
                                if isinstance(mesh, dict) and 'object name' in mesh:
                                    placable_objects.add(mesh['object name'])
                print(f"Loaded {len(placable_objects)} floor placable objects")
            except Exception as e:
                print(f"Warning: Could not load floor placable objects: {e}")
        
        # Also load table placable objects
        table_file = Path("../data/mapping_placable/placable_on_table_list.json")
        if table_file.exists():
            try:
                with open(table_file, 'r', encoding='utf-8') as f:
                    table_data = json.load(f)
                    # Extract object names from nested structure
                    for category, category_data in table_data.items():
                        if isinstance(category_data, dict) and 'meshes' in category_data:
                            for mesh in category_data['meshes']:
                                if isinstance(mesh, dict) and 'object name' in mesh:
                                    placable_objects.add(mesh['object name'])
                print(f"Total placable objects (floor + table): {len(placable_objects)}")
            except Exception as e:
                print(f"Warning: Could not load table placable objects: {e}")
        
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
    
    def _filter_placable_objects(self, objects: List[str]) -> List[str]:
        """Filter objects to only include placable ones"""
        return [obj for obj in objects if obj in self.placable_objects]
    
    def _get_question_types_for_object(self, sm_object: str) -> Set[str]:
        """Get question types that an object can answer"""
        human_name = self.object_mapping.get(sm_object)
        if not human_name:
            return set()
        
        question_types = set()
        for question_type, qa_data in self.question_answer_space.items():
            if qa_data.get('sm_valid_objects') and human_name in qa_data['sm_valid_objects']:
                question_types.add(question_type)
        
        return question_types
    
    def _calculate_qa_potential(self, objects: List[str]) -> Dict[str, Any]:
        """Calculate QA potential for a set of objects"""
        placable_objects = self._filter_placable_objects(objects)
        
        if len(placable_objects) < 2:
            return {
                'placable_count': len(placable_objects),
                'total_qa_potential': 0,
                'question_types_covered': set(),
                'objects_with_qa': []
            }
        
        # Get question types for each placable object
        object_question_types = {}
        all_question_types = set()
        
        for obj in placable_objects:
            q_types = self._get_question_types_for_object(obj)
            object_question_types[obj] = q_types
            all_question_types.update(q_types)
        
        # Calculate total QA potential
        # For each question type, count how many objects can answer it
        total_qa_potential = 0
        objects_with_qa = []
        
        for question_type in all_question_types:
            objects_for_this_qtype = []
            for obj, q_types in object_question_types.items():
                if question_type in q_types:
                    objects_for_this_qtype.append(obj)
            
            if len(objects_for_this_qtype) >= 1:
                # Each object that can answer this question type contributes to QA potential
                total_qa_potential += len(objects_for_this_qtype)
                objects_with_qa.extend(objects_for_this_qtype)
        
        return {
            'placable_count': len(placable_objects),
            'total_qa_potential': total_qa_potential,
            'question_types_covered': all_question_types,
            'objects_with_qa': list(set(objects_with_qa)),
            'object_question_types': object_question_types
        }
    
    def analyze_all_scenes(self) -> Dict[str, Any]:
        """Analyze QA potential for all scenes"""
        results = {}
        total_scenes = len(self.scene_data)
        total_qa_potential = 0
        scenes_with_qa = 0
        
        print(f"Analyzing QA potential for {total_scenes} scenes...")
        
        for scene_name, objects in self.scene_data.items():
            print(f"Processing scene: {scene_name}")
            
            qa_analysis = self._calculate_qa_potential(objects)
            results[scene_name] = qa_analysis
            
            if qa_analysis['total_qa_potential'] > 0:
                scenes_with_qa += 1
                total_qa_potential += qa_analysis['total_qa_potential']
            
            print(f"  Placable objects: {qa_analysis['placable_count']}")
            print(f"  QA potential: {qa_analysis['total_qa_potential']}")
            print(f"  Question types covered: {len(qa_analysis['question_types_covered'])}")
        
        return {
            'summary': {
                'total_scenes': total_scenes,
                'scenes_with_qa_potential': scenes_with_qa,
                'total_qa_potential': total_qa_potential,
                'average_qa_per_scene': total_qa_potential / scenes_with_qa if scenes_with_qa > 0 else 0
            },
            'scene_analysis': results
        }
    
    def save_analysis(self, output_file: Path):
        """Save the analysis results"""
        analysis_results = self.analyze_all_scenes()
        
        # Convert sets to lists for JSON serialization
        for scene_data in analysis_results['scene_analysis'].values():
            scene_data['question_types_covered'] = list(scene_data['question_types_covered'])
            if 'object_question_types' in scene_data:
                # Convert nested sets to lists
                for obj, q_types in scene_data['object_question_types'].items():
                    scene_data['object_question_types'][obj] = list(q_types)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnalysis saved to: {output_file}")
        print(f"Total scenes: {analysis_results['summary']['total_scenes']}")
        print(f"Scenes with QA potential: {analysis_results['summary']['scenes_with_qa_potential']}")
        print(f"Total QA potential: {analysis_results['summary']['total_qa_potential']}")
        print(f"Average QA per scene: {analysis_results['summary']['average_qa_per_scene']:.1f}")

def main():
    analyzer = SceneQAPotentialAnalyzer()
    output_file = Path("scene_qa_potential_analysis.json")
    analyzer.save_analysis(output_file)

if __name__ == "__main__":
    main()
