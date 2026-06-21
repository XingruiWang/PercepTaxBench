#!/usr/bin/env python3
"""
Analyze Question Answer Space

Generates a comprehensive mapping of all possible valid objects from the complete taxonomy
that could answer each type of question. This allows inspection of the complete answer space
for validation and quality assurance.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
import argparse

# Add the QA modules to the path
sys.path.append(str(Path(__file__).parent.parent))

from modules.qa_modules.taxonomy_utils import TaxonomyUtils
from modules.qa_modules.compositional_utils import CompositionalUtils
from modules.qa_modules.filter_utils import STRUCTURAL_OBJECTS_TO_FILTER
# from modules.qa_modules.filter_utils import is_void_cluster  # COMMENTED OUT - using cluster-based filtering


class QuestionAnswerSpaceAnalyzer:
    """Analyze the complete answer space for all question types"""
    
    def __init__(self, taxonomy_dir: str):
        taxonomy_path = Path(taxonomy_dir)
        self.taxonomy_utils = TaxonomyUtils(taxonomy_path)
        self.compositional_utils = CompositionalUtils(self.taxonomy_utils)
        
        # Get all objects from object descriptions (complete list of ~3127 objects)
        self.all_objects = self._get_all_objects_from_descriptions()
        print(f"Loaded {len(self.all_objects)} objects from object descriptions")
        
        # Load SM and OpenImages object lists
        self.sm_objects = self._load_object_set('/path/to/SpatialReasonerDataGen/object_description/results/object_list_final/sm_objects_list.txt')
        self.openimages_objects = self._load_object_set('/path/to/SpatialReasonerDataGen/object_description/results/object_list_final/openimages_objects_list.txt')
        print(f"Loaded {len(self.sm_objects)} SM objects and {len(self.openimages_objects)} OpenImages objects")
    
    def _load_object_set(self, file_path: str) -> Set[str]:
        """Load object names from a text file into a set"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(f"Warning: Could not load object list from {file_path}")
            return set()
    
    def _separate_objects_by_type(self, objects: List[str]) -> Dict[str, List[str]]:
        """Separate objects into SM, OpenImages, and other categories
        
        Note: Void cluster filtering is already applied to all objects in _filter_objects_for_qa.
        This function adds an EXTRA filter for SM objects only: filters out STRUCTURAL_OBJECTS_TO_FILTER
        (background/architectural objects like walls, ceilings, floors) since they are background
        elements in sim images. OpenImages objects are not filtered here.
        """
        sm_objects_list = []
        openimages_objects_list = []

        
        for obj in objects:
            if obj in self.sm_objects:
                # EXTRA filter for SM objects: exclude structural/background objects
                # (void cluster filtering already applied to all objects earlier)
                if obj.lower() not in STRUCTURAL_OBJECTS_TO_FILTER:
                    sm_objects_list.append(obj)
            if obj in self.openimages_objects:  # Changed from elif to if
                # OpenImages objects: only void cluster filtering applied (no structural filter)
                openimages_objects_list.append(obj)
        
        return {
            'sm_objects': sm_objects_list,
            'openimages_objects': openimages_objects_list,
            'sm_count': len(sm_objects_list),
            'openimages_count': len(openimages_objects_list),
        }
    
    def _create_result_entry(self, description: str, filter_criteria: List[str], objects: List[str], 
                           additional_fields: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a standardized result entry with separated SM and OpenImages objects"""
        separated = self._separate_objects_by_type(objects)
        
        result = {
            'description': description,
            'filter_criteria': filter_criteria,
            'total_objects_tested': len(self.all_objects),
            'sm_valid_objects': separated['sm_objects'],
            'openimages_valid_objects': separated['openimages_objects'],
            'sm_objects_count': separated['sm_count'],
            'openimages_objects_count': separated['openimages_count'],
            'sm_percentage': round(separated['sm_count'] / len(objects) * 100, 1) if objects else 0.0,
            'openimages_percentage': round(separated['openimages_count'] / len(objects) * 100, 1) if objects else 0.0
        }
        
        if additional_fields:
            result.update(additional_fields)
        
        return result
    
    def _get_all_objects_from_descriptions(self) -> List[str]:
        """Get all object names from the all_objects_merged.txt file"""
        objects_file = Path("/path/to/SpatialReasonerDataGen/object_description/results/object_list_final/all_objects_merged.txt")
        
        if objects_file.exists():
            with open(objects_file, 'r', encoding='utf-8') as f:
                all_objects = [line.strip() for line in f if line.strip()]
            
            print(f"Loaded {len(all_objects)} objects from all_objects_merged.txt")
            
            # Apply filtering to exclude inappropriate objects
            print(f"Total objects before filtering: {len(all_objects)}")
            filtered_objects = self._filter_objects_for_qa(all_objects)
            print(f"Total objects after filtering: {len(filtered_objects)}")
            return sorted(filtered_objects)
        else:
            print(f"Warning: all_objects_merged.txt file not found: {objects_file}")
            # Fallback to taxonomy objects
            return self._get_all_taxonomy_objects()
    
    def _filter_objects_for_qa(self, objects: List[str]) -> List[str]:
        """Filter objects to exclude inappropriate ones for QA generation using cluster-based filtering"""
        filtered = []
        
        for obj_name in objects:
            
            # Skip objects from specific exclusion clusters
            try:
                should_skip = False
                
                # Overall filter: Only exclude truly inappropriate clusters
                # Check affordance clusters - exclude only human roles (not natural scenes)
                affordance_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_affordances')
                if affordance_clusters:
                    for cluster in affordance_clusters:
                        if cluster in ['Human Roles & Identities (Occupations/Person Types)']:
                            should_skip = True
                            break
                
                # Check function clusters - exclude roles, occupation, and directed actions
                function_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_functions')
                if function_clusters:
                    for cluster in function_clusters:
                        if cluster in ['Roles, Occupation, and Directed Actions']:
                            should_skip = True
                            break
                
                # Check material clusters - exclude abstract/depictions/scenes/occupations
                material_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
                if material_clusters:
                    for cluster in material_clusters:
                        if cluster == 'Abstract / Depictions / Scenes/ Occupations':
                            should_skip = True  # Skip abstract objects
                            break
                
                if should_skip:
                    continue  # Skip this object
                            
            except:
                # If there's an error getting clusters, include the object (safer to include than exclude)
                pass
            
            # If we get here, the object passed all cluster-based filters
            filtered.append(obj_name)
        
        return filtered
    
    def _get_all_taxonomy_objects(self) -> List[str]:
        """Get all object names from the taxonomy"""
        all_objects = set()
        
        # Get objects from all taxonomy types
        taxonomy_types = [
            'final_taxonomy_physical_properties',
            'final_taxonomy_affordances', 
            'final_taxonomy_material',
            'final_taxonomy_function'
        ]
        
        #print(f"DEBUG: Available taxonomy clusters: {list(self.taxonomy_utils.taxonomy_clusters.keys())}")
        
        for tax_type in taxonomy_types:
            taxonomy_data = self.taxonomy_utils.taxonomy_clusters.get(tax_type, {})
            print(f"DEBUG: {tax_type} has {len(taxonomy_data)} top-level keys")
            
            # Handle nested structure (e.g., {"physical_properties": {...}})
            if isinstance(taxonomy_data, dict) and len(taxonomy_data) == 1:
                inner_key = list(taxonomy_data.keys())[0]
                clusters = taxonomy_data[inner_key]
                print(f"DEBUG: Using inner key '{inner_key}' for {tax_type}")
            else:
                clusters = taxonomy_data
            
            # Extract objects from cluster data
            for cluster_name, cluster_data in clusters.items():
                if isinstance(cluster_data, dict) and 'objects' in cluster_data:
                    all_objects.update(cluster_data['objects'])
                elif isinstance(cluster_data, list):
                    all_objects.update(cluster_data)
        
        return sorted(list(all_objects))
    
    def analyze_all_question_types(self) -> Dict[str, Dict[str, Any]]:
        """Analyze answer space for all question types"""
        results = {}
        
        results.update(self._analyze_compositional_questions())
        
        results.update(self._analyze_functional_questions())
        
        results.update(self._analyze_material_questions())
        results.update(self._analyze_affordance_questions())
        results.update(self._analyze_repurposing_questions())
        results.update(self._analyze_material_inference_questions())
        results.update(self._analyze_description_based_questions())
        results.update(self._analyze_counterfactual_questions())
        results.update(self._analyze_latent_state_questions())
        
        return results
    
    def _analyze_compositional_questions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze compositional question types"""
        results = {}
        
        compositional_templates = [
            {
                'type': 'compositional_set_subtraction_hollow',
                'required': ['hollow', 'rigid'],
                'forbidden': ['contain'],
                'description': 'hollow and rigid but NOT designed as a container'
            },
            {
                'type': 'compositional_set_subtraction_container',
                'required': ['rigid', 'movable'],
                'forbidden': ['contain', 'hollow'],
                'description': 'rigid, movable, but NOT a container'
            }
        ]
        
        for template in compositional_templates:
            valid_objects = []
            
            for obj_name in self.all_objects:
                try:
                    is_valid = self.compositional_utils.check_properties(
                        obj_name.lower(),
                        template['required'],
                        template['forbidden']
                    )
                    if is_valid:
                        valid_objects.append(obj_name)
                except Exception as e:
                    pass
            
            results[template['type']] = self._create_result_entry(
                template['description'],
                template['required'] + template['forbidden'],
                sorted(valid_objects),
                {
                    'required_properties': template['required'],
                    'forbidden_properties': template['forbidden']
                }
            )
        
        return results
    
    def _analyze_functional_questions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze functional question types"""
        results = {}
        
        
        affordance_taxonomy = self.taxonomy_utils.taxonomy_clusters.get('final_taxonomy_affordances', {})
        if isinstance(affordance_taxonomy, dict) and 'affordance' in affordance_taxonomy:
            affordance_clusters = affordance_taxonomy['affordance']
            seating_cluster_data = affordance_clusters.get('Sit / Ride / Attend', {})
            seating_objects = seating_cluster_data.get('objects', []) if isinstance(seating_cluster_data, dict) else []
        else:
            seating_objects = []
        
        results['functional_seating'] = {
            'description': 'objects designed for sitting or resting',
            'filter_criteria': ['Sit / Ride / Attend affordance cluster'],
            'total_objects_tested': len(self.all_objects),
            'valid_objects': sorted(seating_objects),
            'valid_count': len(seating_objects)
        }
        
        
        foldable_objects_set = set()
        
       
        foldable_material_clusters = [
            'Textiles, Fibers & Leather',
            'Plastics, Rubber & Polymers'
        ]
        
        for obj_name in self.all_objects:
            # First check: Must be in Flexible physical property cluster
            if not self.taxonomy_utils.has_property(obj_name, 'flexible'):
                continue
            
            # Check material clusters for foldable materials
            clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            if clusters:
                if any(cluster in clusters for cluster in foldable_material_clusters):
                    foldable_objects_set.add(obj_name)
            
            # Also include keyword-based matches for specific foldable items
            obj_lower = obj_name.lower()
            foldable_keywords = ['fold', 'collapsible', 'retractable', 'foldable', 'folding', 'blanket', 'towel', 'cloth', 'curtain', 'shirt', 'clothing', 'pillow', 'rug', 'sheet']
            if any(keyword in obj_lower for keyword in foldable_keywords):
                foldable_objects_set.add(obj_name)
        
        foldable_objects = sorted(list(foldable_objects_set))
        
        results['functional_foldable'] = {
            'description': 'objects that can be folded or collapsed (clothing, fabrics, flexible materials)',
            'filter_criteria': [
                'MUST be in Flexible physical property cluster',
                'Textiles, Fibers & Leather cluster',
                'Plastics, Rubber & Polymers cluster', 
                'INCLUDE: clothing, fabrics, flexible materials',
                'INCLUDE: items that can be folded, compressed, or manipulated'
            ],
            'total_objects_tested': len(self.all_objects),
            'valid_objects': sorted(foldable_objects),
            'valid_count': len(foldable_objects)
        }
        
        return results
    
    def _analyze_material_questions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze material-based question types"""
        results = {}
        
        # Flammable objects
        flammable_clusters = ['Textiles, Fibers & Leather', 'Paper, Cardboard & Pulp', 'Plastics, Rubber & Polymers']
        flammable_objects = []
        
        print(f"DEBUG: Testing {len(self.all_objects)} objects for flammability")
        print(f"DEBUG: First 10 objects in self.all_objects: {self.all_objects[:10]}")
        
        # Check what's actually in the textiles cluster
        textiles_cluster_objects = self.taxonomy_utils.taxonomy_clusters.get('final_taxonomy_material', {}).get('material', {}).get('Textiles, Fibers & Leather', {}).get('objects', [])
        print(f"DEBUG: First 10 objects in textiles cluster: {textiles_cluster_objects[:10]}")
        print(f"DEBUG: Total objects in textiles cluster: {len(textiles_cluster_objects)}")
        
        textiles_found = 0
        
        for obj_name in self.all_objects:
            clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            if clusters:
                if 'Textiles, Fibers & Leather' in clusters:
                    textiles_found += 1
                    if textiles_found <= 5:  # Print first 5 textiles objects found
                        print(f"DEBUG: Found textiles object: {obj_name}")
                
                
                affordance_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_affordances')
                should_exclude_for_flammability = False
                if affordance_clusters:
                    for cluster in affordance_clusters:
                        if cluster in ['Natural Scenes (View/Appraise)', 'Phenomena (View/Read/Appraise)']:
                            should_exclude_for_flammability = True
                            break
                
                if not should_exclude_for_flammability and any(cluster in clusters for cluster in flammable_clusters):
                    flammable_objects.append(obj_name)
        
        print(f"DEBUG: Found {textiles_found} textiles objects, {len(flammable_objects)} total flammable objects")
        
        results['flammability'] = {
            'description': 'objects that pose fire risk (flammable materials)',
            'filter_criteria': flammable_clusters,
            'total_objects_tested': len(self.all_objects),
            'valid_objects': sorted(flammable_objects),
            'valid_count': len(flammable_objects)
        }
        
        # Analyze by specific material clusters
        material_taxonomy = self.taxonomy_utils.taxonomy_clusters.get('final_taxonomy_material', {})
        
        # Handle nested structure
        if isinstance(material_taxonomy, dict) and 'material' in material_taxonomy:
            material_clusters = material_taxonomy['material']
        else:
            material_clusters = material_taxonomy
            
        for cluster_name, cluster_data in material_clusters.items():
            if isinstance(cluster_data, dict) and 'objects' in cluster_data:
                
                inappropriate_material_clusters = [
                    'Abstract / Depictions / Scenes/ Occupations', 
                    'Unclassified'  
                ]
                if cluster_name in inappropriate_material_clusters:
                    continue
                    
                objects = cluster_data['objects']
                # Clean up cluster name for question type
                clean_name = cluster_name.lower().replace(" ", "_").replace(",", "").replace("&", "and").replace("__", "_").replace("‑", "_")
                results[f'material_{clean_name}'] = {
                    'description': f'objects with {cluster_name} material properties',
                    'filter_criteria': [cluster_name],
                    'total_objects_tested': len(self.all_objects),
                    'valid_objects': sorted(objects),
                    'valid_count': len(objects)
                }
        
        return results
    
    def _analyze_affordance_questions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze affordance-based question types"""
        results = {}
        
        affordance_taxonomy = self.taxonomy_utils.taxonomy_clusters.get('final_taxonomy_affordances', {})
        
        # Handle nested structure
        if isinstance(affordance_taxonomy, dict) and 'affordance' in affordance_taxonomy:
            affordance_clusters = affordance_taxonomy['affordance']
        else:
            affordance_clusters = affordance_taxonomy
            
        for cluster_name, cluster_data in affordance_clusters.items():
            if isinstance(cluster_data, dict) and 'objects' in cluster_data:
                
                inappropriate_affordance_clusters = [
                    'Human Roles & Identities (Occupations/Person Types)',  # People/occupations
                    'Natural Scenes (View/Appraise)',  # Scenes/environments
                    'Phenomena (View/Read/Appraise)',  # Abstract phenomena
                    'Unclassified'  # Objects without proper categorization
                ]
                if cluster_name in inappropriate_affordance_clusters:
                    continue
                    
                objects = cluster_data['objects']

                clean_name = cluster_name.lower().replace(" ", "_").replace("/", "_").replace("&", "and").replace("__", "_").replace("‑", "_")
                results[f'affordance_{clean_name}'] = {
                    'description': f'objects with {cluster_name} affordances',
                    'filter_criteria': [cluster_name],
                    'total_objects_tested': len(self.all_objects),
                    'valid_objects': sorted(objects),
                    'valid_count': len(objects)
                }
        
        return results
    
    def _analyze_repurposing_questions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze repurposing question types using updated filtering logic with separate property clusters"""
        results = {}
        
        # Define repurposing concepts 
        repurposing_concepts = {
            'shield_concept': {
                'name': 'Shield Concept',
                'new_purpose': 'Protection/Blocking',
                'description': 'objects that could be repurposed as shields (flat, rigid, stable, movable, width/length > height, not fragile, not food/electronics/furniture/large architectural items)'
            },
            'container_concept': {
                'name': 'Container Concept',
                'new_purpose': 'Storage/Holding',
                'description': 'objects that could be repurposed as containers (hollow or container affordance)'
            },
            'reflector_concept': {
                'name': 'Reflector Concept',
                'new_purpose': 'Light Reflection/Signaling',
                'description': 'objects that could be repurposed as reflectors (glossy texture, smooth surfaces, or metallic/glass/ceramic materials)'
            },
            'cushion_concept': {
                'name': 'Cushion Concept',
                'new_purpose': 'Comfort/Protection',
                'description': 'objects that could be repurposed as cushions (soft, flexible, padded materials)'
            },
            'stepstool_concept': {
                'name': 'Step Stool Concept',
                'new_purpose': 'Elevation/Reaching',
                'description': 'objects that could be repurposed as step stools (rigid, stable, flat surfaces, movable)'
            },
            'bookend_concept': {
                'name': 'Bookend Concept',
                'new_purpose': 'Organization/Support',
                'description': 'objects that could be repurposed as bookends (rigid, stable, upright, movable)'
            },
        }
        
        # Analyze each repurposing concept using updated filtering logic
        for concept_key, concept_data in repurposing_concepts.items():
            valid_objects = []
            
            for obj_name in self.all_objects:
                try:
                    if self._object_matches_repurposing_concept(obj_name, concept_key):
                        valid_objects.append(obj_name)
                except Exception as e:
                    # Skip objects that cause errors during property checking
                    pass
            
            # Set specific filter criteria based on concept type
            if concept_key == 'tool_concept':
                filter_criteria = [
                    'REPURPOSING: (rigid AND durable AND movable AND NOT flexible AND NOT unstable AND NOT fixed)',
                    'EXCLUDE: Objects already in "Defense, Weapons, Enforcement & Tools" function cluster',
                    'EXCLUDE: Obvious existing tools (hammer, ruler, calculator, camera, etc.)',
                    'EXCLUDE: Abstract, Biological, Food, Liquids, Gases, Textiles materials',
                    'EXCLUDE: Living things, Food, Architectural, Furniture, Large appliances, Vehicles affordances',
                    'EXCLUDE: Large/stationary objects (appliances, furniture, buildings, vehicles)'
                ]
            elif concept_key == 'shield_concept':
                filter_criteria = [
                    'rigid AND stable AND movable AND NOT fragile',
                    'EXCLUDE: Operate / Use Device affordance',
                    'EXCLUDE: Mechanical Control affordance',
                    'EXCLUDE: Build / Span / Occupy affordance',
                    'EXCLUDE: Sit / Ride / Attend affordance',
                    'EXCLUDE: Control / Express / Light affordance',
                    'EXCLUDE: Household / Facility Operations affordance',
                    'EXCLUDE: Architectural Components & Fixtures affordance',
                    'EXCLUDE: Enclosures & Venues (Enter/Use) affordance',
                    'EXCLUDE: Food — Ingredients & Produce affordance',
                    'EXCLUDE: Grow / Plant (Vegetation) affordance',
                    'EXCLUDE: Tableware and Serveware affordance',
                    'EXCLUDE: Grip / Carry / Operate affordance',
                    'EXCLUDE: Medical & Healthcare Equipment function',
                    'EXCLUDE: Devices, Displays, Transactions & Admin function',
                    'EXCLUDE: Architecture, Rooms & Built Spaces function',
                    'EXCLUDE: Vehicles, Transportation function',
                    'EXCLUDE: Play, Sports, Toys & Performance function',
                    'EXCLUDE: Furniture, Storage, Stores & Interiors function',
                    'EXCLUDE: Viewing, Presentation & Signage function',
                    'EXCLUDE: Kitchen Ecosystem function',
                    'EXCLUDE: Containers, Vessels, Bags & Holding function',
                    'EXCLUDE: Defense, Weapons, Enforcement & Tools function',
                    'EXCLUDE: Protection, Coverings, Curtains & Barriers function',
                    'EXCLUDE: Cleaning and Sanitation function',
                    'EXCLUDE: Food & Drink function',
                    'EXCLUDE: Botanical, Wildlife & Environmental Support function'
                ]
            elif concept_key == 'container_concept':
                filter_criteria = [
                    'hollow property OR Contain/Carry/Package affordance',
                    'EXCLUDE: Medical & Healthcare Equipment function',
                    'EXCLUDE: Devices, Displays, Transactions & Admin function',
                    'EXCLUDE: Architecture, Rooms & Built Spaces function', 
                    'EXCLUDE: Vehicles, Transportation function',
                    'EXCLUDE: Play, Sports, Toys & Performance function',
                    'EXCLUDE: Operate / Use Device affordance',
                    'EXCLUDE: Mechanical Control affordance',
                    'EXCLUDE: Build / Span / Occupy affordance',
                    'EXCLUDE: Sit / Ride / Attend affordance',
                    'EXCLUDE: Control / Express / Light affordance',
                    'EXCLUDE: Household / Facility Operations affordance',
                    'EXCLUDE: Architectural Components & Fixtures affordance'
                ]
            elif concept_key == 'reflector_concept':
                filter_criteria = [
                    'Liquid physical property OR smooth OR (Metals/Glass materials)',
                    'INCLUDE: Liquid objects in Outdoor Environments (lakes, rivers, streams, etc.)',
                    'EXCLUDE: Operate / Use Device affordance',
                    'EXCLUDE: Mechanical Control affordance',
                    'EXCLUDE: Build / Span / Occupy affordance',
                    'EXCLUDE: Sit / Ride / Attend affordance',
                    'EXCLUDE: Control / Express / Light affordance',
                    'EXCLUDE: Household / Facility Operations affordance',
                    'EXCLUDE: Architectural Components & Fixtures affordance',
                    'EXCLUDE: Enclosures & Venues (Enter/Use) affordance',
                    'EXCLUDE: Food — Ingredients & Produce affordance',
                    'EXCLUDE: Grow / Plant (Vegetation) affordance',
                    'EXCLUDE: Tableware and Serveware affordance',
                    'EXCLUDE: Grip / Carry / Operate affordance',
                    'EXCLUDE: Medical & Healthcare Equipment function',
                    'EXCLUDE: Devices, Displays, Transactions & Admin function',
                    'EXCLUDE: Architecture, Rooms & Built Spaces function',
                    'EXCLUDE: Vehicles, Transportation function',
                    'EXCLUDE: Play, Sports, Toys & Performance function',
                    'EXCLUDE: Furniture, Storage, Stores & Interiors function',
                    'EXCLUDE: Viewing, Presentation & Signage function',
                    'EXCLUDE: Kitchen Ecosystem function',
                    'EXCLUDE: Containers, Vessels, Bags & Holding function',
                    'EXCLUDE: Defense, Weapons, Enforcement & Tools function',
                    'EXCLUDE: Protection, Coverings, Curtains & Barriers function',
                    'EXCLUDE: Cleaning and Sanitation function',
                    'EXCLUDE: Food & Drink function',
                    'EXCLUDE: Botanical, Wildlife & Environmental Support function'
                ]
            elif concept_key == 'cushion_concept':
                filter_criteria = ['flexible property OR Textiles materials']
            elif concept_key == 'stepstool_concept':
                filter_criteria = [
                    'rigid AND stable AND movable',
                    'EXCLUDE: Operate / Use Device affordance',
                    'EXCLUDE: Mechanical Control affordance',
                    'EXCLUDE: Build / Span / Occupy affordance',
                    'EXCLUDE: Sit / Ride / Attend affordance',
                    'EXCLUDE: Control / Express / Light affordance',
                    'EXCLUDE: Household / Facility Operations affordance',
                    'EXCLUDE: Architectural Components & Fixtures affordance',
                    'EXCLUDE: Food — Ingredients & Produce affordance',
                    'EXCLUDE: Furniture affordance',
                    'EXCLUDE: Display / Exhibit / Signal Value affordance',
                    'EXCLUDE: Enclosures & Venues (Enter/Use) affordance',
                    'EXCLUDE: Medical & Healthcare Equipment function',
                    'EXCLUDE: Devices, Displays, Transactions & Admin function',
                    'EXCLUDE: Architecture, Rooms & Built Spaces function',
                    'EXCLUDE: Vehicles, Transportation function',
                    'EXCLUDE: Play, Sports, Toys & Performance function',
                    'EXCLUDE: Food & Drink function',
                    'EXCLUDE: Furniture, Storage, Stores & Interiors function',
                    'EXCLUDE: Viewing, Presentation & Signage function',
                    'EXCLUDE: Kitchen Ecosystem function',
                    'EXCLUDE: Containers, Vessels, Bags & Holding function',
                    'EXCLUDE: Defense, Weapons, Enforcement & Tools function',
                    'EXCLUDE: Protection, Coverings, Curtains & Barriers function',
                    'EXCLUDE: Cleaning and Sanitation function'
                ]
            elif concept_key == 'bookend_concept':
                filter_criteria = ['rigid AND stable AND movable']
            elif concept_key == 'doorstop_concept':
                filter_criteria = [
                    'heavy AND stable AND movable',
                    'EXCLUDE: Operate / Use Device affordance',
                    'EXCLUDE: Mechanical Control affordance',
                    'EXCLUDE: Build / Span / Occupy affordance',
                    'EXCLUDE: Sit / Ride / Attend affordance',
                    'EXCLUDE: Control / Express / Light affordance',
                    'EXCLUDE: Household / Facility Operations affordance',
                    'EXCLUDE: Architectural Components & Fixtures affordance',
                    'EXCLUDE: Food — Ingredients & Produce affordance',
                    'EXCLUDE: Furniture affordance',
                    'EXCLUDE: Display / Exhibit / Signal Value affordance',
                    'EXCLUDE: Enclosures & Venues (Enter/Use) affordance',
                    'EXCLUDE: Medical & Healthcare Equipment function',
                    'EXCLUDE: Devices, Displays, Transactions & Admin function',
                    'EXCLUDE: Architecture, Rooms & Built Spaces function',
                    'EXCLUDE: Vehicles, Transportation function',
                    'EXCLUDE: Play, Sports, Toys & Performance function',
                    'EXCLUDE: Food & Drink function',
                    'EXCLUDE: Furniture, Storage, Stores & Interiors function',
                    'EXCLUDE: Viewing, Presentation & Signage function',
                    'EXCLUDE: Kitchen Ecosystem function',
                    'EXCLUDE: Containers, Vessels, Bags & Holding function',
                    'EXCLUDE: Defense, Weapons, Enforcement & Tools function',
                    'EXCLUDE: Protection, Coverings, Curtains & Barriers function',
                    'EXCLUDE: Cleaning and Sanitation function'
                ]
            elif concept_key == 'magnifying_glass_concept':
                filter_criteria = ['stable AND movable AND smooth AND rigid AND (Glass/Plastic materials)']
            else:
                filter_criteria = [f'Updated filtering using separate property clusters']
            
            results[f'repurposing_{concept_key}'] = {
                'description': concept_data['description'],
                'filter_criteria': filter_criteria,
                'concept_name': concept_data['name'],
                'new_purpose': concept_data['new_purpose'],
                'total_objects_tested': len(self.all_objects),
                'valid_objects': sorted(valid_objects)
            }
        
        return results
    
    def _object_matches_repurposing_concept(self, obj_name: str, concept_key: str) -> bool:
        """Check if object matches repurposing concept using updated separate property clusters"""
        
        if concept_key == "tool_concept":
            
            has_rigid = self.taxonomy_utils.has_property(obj_name, 'rigid')
            has_durable = self.taxonomy_utils.has_property(obj_name, 'durable')
            has_flexible = self.taxonomy_utils.has_property(obj_name, 'flexible')
            has_unstable = self.taxonomy_utils.has_property(obj_name, 'unstable')
            
            
            material_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            affordance_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_affordances')
            function_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_function')
            
           
            has_tool_affordance = 'Grip / Carry / Operate' in affordance_clusters
            is_already_tool = 'Defense, Weapons, Enforcement & Tools' in function_clusters
            
            
            obvious_tool_names = [
                'tool', 'hammer', 'screwdriver', 'wrench', 'knife', 'scissors', 'pliers', 'drill',
                'ruler', 'eraser', 'binder', 'clip', 'calculator', 'camera', 'metal detector'
            ]
            is_obvious_tool = any(tool_name in obj_name.lower() for tool_name in obvious_tool_names)
            
            exclude_material_clusters = [
                'Abstract / Depictions / Scenes/ Occupations',
                'Biological (Animals/Body Parts)', 
                'Biological (Plants/Flowers)',
                'Organic Food & Edible Matter',
                'Liquids & Semi‑liquids',
                'Gases, Vapors & Atmospheric',
                'Textiles, Fibers & Leather'  
            ]
            
            is_in_excluded_material = any(cluster in material_clusters for cluster in exclude_material_clusters)
            
            
            exclude_affordance_clusters = [
                'Interact with Living/Moving Things',
                'Food — Ingredients & Produce', 
                'Human Roles & Identities (Occupations/Person Types)',
                'Food — Prepared Dishes',
                'Build / Span / Occupy',  
                'Enclosures & Venues (Enter/Use)',  
                'Natural Scenes (View/Appraise)',  
                'Art Display (View/Appraise)',  
                'Phenomena (View/Read/Appraise)',  
                'Sit / Ride / Attend',  
                'Architectural Components and Fixtures',  
                'Household / Facility Operations'  
            ]
            
            is_in_excluded_affordance = any(cluster in affordance_clusters for cluster in exclude_affordance_clusters)
            
            
            hand_held_exclusions = [
                'air conditioner', 'atm', 'appliance', 'amplifier', 'air vent', 'ac vent',
                'armchair', 'bed', 'bench', 'chair', 'couch', 'desk', 'table', 'sofa', 'stool',
                'refrigerator', 'fridge', 'freezer', 'oven', 'stove', 'washing machine',
                'television', 'tv', 'computer', 'monitor', 'screen', 'projector',
                'building', 'wall', 'roof', 'ceiling', 'floor', 'door', 'window',
                'car', 'vehicle', 'truck', 'bus', 'boat', 'ship', 'aircraft',
                'hoop', 'basketball', 'sports', 'field', 'court', 'arena'
            ]
            
            is_large_stationary = any(exclusion in obj_name.lower() for exclusion in hand_held_exclusions)
            
            
            if is_in_excluded_material or is_in_excluded_affordance or is_large_stationary or is_already_tool or is_obvious_tool:
                return False
            
            has_movable = self.taxonomy_utils.has_property(obj_name, 'movable')
            has_fixed = self.taxonomy_utils.has_property(obj_name, 'fixed')
            
            
            return has_rigid and has_durable and not has_flexible and not has_unstable and has_movable and not has_fixed
            
        elif concept_key == "shield_concept":
            # Shields should be flat, rigid, stable, movable, appropriate size, and NOT small tools/utensils/food/electronics
            has_rigid = self.taxonomy_utils.has_property(obj_name, 'rigid')
            has_stable = self.taxonomy_utils.has_property(obj_name, 'stable')
            has_movable = self.taxonomy_utils.has_property(obj_name, 'movable')
            
            # Check flatness requirements
            has_flatness = self._check_flatness_for_shield(obj_name)
            
            # Check size requirements
            has_appropriate_size = self._check_shield_size(obj_name)
            
            # Get clusters for systematic filtering
            affordance_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_affordances')
            function_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_function')
            
            # EXCLUDE inappropriate objects using affordance clusters
            exclude_affordance_clusters = [
                'Build / Span / Occupy',           # Infrastructure
                'Sit / Ride / Attend',             # Transportation
                'Household / Facility Operations', # Exercise equipment
                'Architectural Components & Fixtures',  # Infrastructure components
                'Enclosures & Venues (Enter/Use)', # Large venues/equipment
                'Food — Ingredients & Produce',   # Food items
                'Grow / Plant (Vegetation)'        # Plants/vegetation
            ]
            is_in_excluded_affordance = any(cluster in affordance_clusters for cluster in exclude_affordance_clusters)
            
            # EXCLUDE inappropriate objects using function clusters
            exclude_function_clusters = [
                'Medical & Healthcare Equipment',
                'Devices, Displays, Transactions & Admin', 
                'Architecture, Rooms & Built Spaces',
                'Vehicles, Transportation ',
                'Furniture, Storage, Stores & Interiors',  # Large furniture
                'Defense, Weapons, Enforcement & Tools',  # Large safety equipment
                'Protection, Coverings, Curtains & Barriers',  # Large protective equipment
                'Food & Drink',  # Food items
                'Botanical, Wildlife & Environmental Support'  # Plants/vegetation
            ]
            is_in_excluded_function = any(cluster in function_clusters for cluster in exclude_function_clusters)
            
            # EXCLUDE objects that are fragile (not suitable for protection)
            has_fragile = self.taxonomy_utils.has_property(obj_name, 'fragile')
            
            return (has_rigid and has_stable and has_movable and has_flatness and 
                   has_appropriate_size and not is_in_excluded_affordance and 
                   not is_in_excluded_function and not has_fragile)
            
        elif concept_key == "container_concept":
            # Containers should be hollow or have container affordance - use cluster-based filtering
            has_hollow = self.taxonomy_utils.has_property(obj_name, 'hollow')
            
            # Check container affordance cluster
            affordance_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_affordances')
            has_container_affordance = 'Contain / Carry / Package' in affordance_clusters
            
            # Get function clusters for systematic filtering
            function_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_function')
            
            # EXCLUDE objects that are primarily equipment, infrastructure using function clusters
            exclude_function_clusters = [
                'Medical & Healthcare Equipment',
                'Devices, Displays, Transactions & Admin', 
                'Architecture, Rooms & Built Spaces',
                'Vehicles, Transportation '
            ]
            is_in_excluded_function = any(cluster in function_clusters for cluster in exclude_function_clusters)
            
            # EXCLUDE objects that are primarily infrastructure using affordance clusters
            exclude_affordance_clusters = [
                'Operate / Use Device', 
                'Mechanical Control', 
                'Build / Span / Occupy',           
                'Sit / Ride / Attend',             
                'Control / Express / Light',      
                'Household / Facility Operations', 
                'Architectural Components & Fixtures' 
            ]
            is_in_excluded_affordance = any(cluster in affordance_clusters for cluster in exclude_affordance_clusters)
            
            # Only include if it has container properties AND is not equipment/infrastructure
            return (has_hollow or has_container_affordance) and not is_in_excluded_function and not is_in_excluded_affordance
        
        
        elif concept_key == "reflector_concept":
            # Check if object is in Liquid physical property cluster
            is_liquid = self.taxonomy_utils.has_property(obj_name, 'liquid')
            
            # Reflectors should have glossy/reflective surfaces - use texture taxonomy for precision
            texture_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_texture')
            has_glossy_texture = 'Mirror/Glossy Hard Surfaces (Metal/Glass/Ceramic)' in texture_clusters
            
            # Also check for smooth physical property as backup
            has_smooth = self.taxonomy_utils.has_property(obj_name, 'smooth')
            
            # Check for reflective materials (metals, glass, ceramics)
            material_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            is_reflective_material = any(cluster in material_clusters for cluster in [
                'Metals & Alloys', 
                'Glass & Transparent (Silicate)', 
                'Ceramics, Porcelain & Earthenware'
            ])
            
           
            function_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_function')
            exclude_function_clusters = [
                'Food & Drink ',  
                'Roles, Occupation, and Directed Actions', 
                'Architecture, Rooms & Built Spaces',  
            ]
            
            exclude_outdoor_cluster = 'Outdoor Environments, Terrain & Circulation' in function_clusters
            if exclude_outdoor_cluster and not is_liquid:
                exclude_function_clusters.append('Outdoor Environments, Terrain & Circulation')
            
            is_in_excluded_function = any(cluster in function_clusters for cluster in exclude_function_clusters)
            
            # Include liquids OR (glossy texture OR (smooth AND reflective material)), AND exclude inappropriate objects
            return (is_liquid or has_glossy_texture or (has_smooth and is_reflective_material)) and not is_in_excluded_function
        
        elif concept_key == "cushion_concept":
            # Cushions should be flexible, durable, movable - use proper physical properties
            has_flexible = self.taxonomy_utils.has_property(obj_name, 'flexible')
            has_durable = self.taxonomy_utils.has_property(obj_name, 'durable')
            has_movable = self.taxonomy_utils.has_property(obj_name, 'movable')
            
            # Check for soft/padded textures
            texture_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_texture')
            has_soft_texture = any(cluster in texture_clusters for cluster in [
                'Soft/Elastic Polymers (Rubber/Foam/Soft Plastics)',
                'Textiles & Clothing (Fabric/Garments/Cloth)'
            ])
            
            # EXCLUDE hard textures (cushions should not be hard)
            has_hard_texture = any(cluster in texture_clusters for cluster in [
                'Hard Smooth (Non-Glossy Smooth Surfaces)',
                'Mirror/Glossy Hard Surfaces (Metal/Glass/Ceramic)',
                'Matte/Opaque Hard Surfaces (Painted/Stone/Plastic/wooden)',
                'Geologic/Masonry (Rock/Brick/Concrete/Soil)'
            ])
            
            # Check for soft materials
            material_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            is_soft_material = any(cluster in material_clusters for cluster in [
                'Textiles, Fibers & Leather',
                'Plastics, Rubber & Polymers'
            ])
            
            # EXCLUDE inappropriate materials (hard materials, liquids, gases)
            has_inappropriate_material = any(cluster in material_clusters for cluster in [
                'Wood & Plant‑Based Solids',  
                'Biological (Plants/Flowers)',  
                'Biological (Animals/Body Parts)',  
                'Metals & Alloys',  
                'Stone, Concrete & Mineral',  
                'Glass & Transparent (Silicate)',  
                'Ceramics, Porcelain & Earthenware', 
                'Liquids & Semi-Liquids', 
                'Gases, Vapors & Atmospheric' 
            ])
            
            # EXCLUDE food/drink clusters using function clusters
            function_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_function')
            exclude_food_function_clusters = [
                'Food & Drink '  # Food and drink items (note the trailing space)
            ]
            is_food_function = any(cluster in function_clusters for cluster in exclude_food_function_clusters)
            
            # EXCLUDE clothing items using affordance clusters
            affordance_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_affordances')
            exclude_clothing_affordance_clusters = [
                'Wearables & Apparel'  
            ]
            is_clothing_affordance = any(cluster in affordance_clusters for cluster in exclude_clothing_affordance_clusters)
            
            
            soft_clothing_keywords = ['blanket', 'towel', 'scarf', 'shawl', 'wrap', 'robe', 'sweater', 'hoodie', 'cardigan', 'pullover', 'pajamas', 'nightgown', 'kimono', 'sarong', 'poncho', 'cape', 'curtain', 'drape', 'fabric', 'cloth', 'textile']
            is_soft_clothing = any(keyword in obj_name.lower() for keyword in soft_clothing_keywords)
            
            
            if is_clothing_affordance and not is_soft_clothing:
                is_clothing_affordance = True  
            elif is_clothing_affordance and is_soft_clothing:
                is_clothing_affordance = False  #
            
            # Must have required physical properties AND positive criteria AND no negative criteria
            has_required_properties = has_flexible and has_durable and has_movable
            has_positive_criterion = has_soft_texture or is_soft_material
            
            return (has_required_properties and 
                   has_positive_criterion and
                   not has_hard_texture and 
                   not has_inappropriate_material and
                   not is_food_function and
                   not is_clothing_affordance)
        
        elif concept_key == "stepstool_concept":
            # CLUSTER-BASED: Use Heavy + Rigid + Stable clusters + exclude specific clusters
            # Must be heavy, rigid, and stable
            has_heavy = self.taxonomy_utils.has_property(obj_name, 'heavy')
            has_rigid = self.taxonomy_utils.has_property(obj_name, 'rigid')
            has_stable = self.taxonomy_utils.has_property(obj_name, 'stable')
            has_movable = self.taxonomy_utils.has_property(obj_name, 'movable')
            
            # Must NOT be flexible, unstable, or fragile
            has_flexible = self.taxonomy_utils.has_property(obj_name, 'flexible')
            has_unstable = self.taxonomy_utils.has_property(obj_name, 'unstable')
            has_fragile = self.taxonomy_utils.has_property(obj_name, 'fragile')
            
            # Get clusters for systematic filtering
            function_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_function')
            
            # EXCLUDE using cluster-based filtering:
            # 1. Food items - use function clusters
            food_function_clusters = ['Food & Drink ']
            is_food = any(cluster in function_clusters for cluster in food_function_clusters)
            
            # 2. Vehicles - use function clusters
            vehicle_function_clusters = ['Vehicles, Transportation ']
            is_vehicle = any(cluster in function_clusters for cluster in vehicle_function_clusters)
            
            # 3. People/Animals - use function clusters
            people_function_clusters = ['Roles, Occupation, and Directed Actions']
            is_people = any(cluster in function_clusters for cluster in people_function_clusters)
            
            # 4. Spaces/Venues - use function clusters
            space_function_clusters = ['Architecture, Rooms & Built Spaces', 'Outdoor Environments, Terrain & Circulation']
            is_space = any(cluster in function_clusters for cluster in space_function_clusters)
            
            # 5. Additional exclusions for stepstool concept
            additional_exclude_clusters = [
                'Furniture, Storage, Stores & Interiors', 
                'Kitchen Ecosystem',  
                'Viewing, Presentation & Signage',  
                'Play, Sports, Toys & Performance', 
                'Medical & Healthcare Equipment',  
                'Containers, Vessels, Bags & Holding',  
                'Defense, Weapons, Enforcement & Tools',  
                'Protection, Coverings, Curtains & Barriers',  
                'Cleaning and Sanitation'  
            ]
            is_additional_excluded = any(cluster in function_clusters for cluster in additional_exclude_clusters)
            
            # Must be heavy, rigid, stable, movable AND not excluded by clusters
            return (has_heavy and has_rigid and has_stable and has_movable and 
                   not has_flexible and not has_unstable and not has_fragile and
                   not is_food and not is_vehicle and not is_people and not is_space and not is_additional_excluded)
        
        elif concept_key == "bookend_concept":
            # Bookends should be heavy, stable, movable, and NOT large equipment/devices/food/furniture
            has_heavy = self.taxonomy_utils.has_property(obj_name, 'heavy')
            has_stable = self.taxonomy_utils.has_property(obj_name, 'stable')
            has_movable = self.taxonomy_utils.has_property(obj_name, 'movable')
            
            # Get clusters for systematic filtering
            affordance_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_affordances')
            function_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_function')
            
            # EXCLUDE large equipment/devices, food, and furniture using affordance clusters
            exclude_affordance_clusters = [
                'Operate / Use Device', 
                'Mechanical Control',  
                'Build / Span / Occupy',           
                'Sit / Ride / Attend',             
                'Control / Express / Light',       
                'Household / Facility Operations', 
                'Architectural Components & Fixtures', 
                'Food — Ingredients & Produce',   
                'Furniture',                      
                'Display / Exhibit / Signal Value', 
                'Enclosures & Venues (Enter/Use)' 
            ]
            is_in_excluded_affordance = any(cluster in affordance_clusters for cluster in exclude_affordance_clusters)
            
            # EXCLUDE large equipment/devices, food, and furniture using function clusters
            exclude_function_clusters = [
                'Medical & Healthcare Equipment',
                'Devices, Displays, Transactions & Admin', 
                'Architecture, Rooms & Built Spaces',
                'Vehicles, Transportation ',
                'Play, Sports, Toys & Performance',
                'Food & Drink ',  
                'Furniture, Storage, Stores & Interiors',  
                'Viewing, Presentation & Signage',  
                'Kitchen Ecosystem', 
                'Containers, Vessels, Bags & Holding',  
                'Defense, Weapons, Enforcement & Tools',  
                'Protection, Coverings, Curtains & Barriers',  
                'Cleaning and Sanitation'
            ]
            is_in_excluded_function = any(cluster in function_clusters for cluster in exclude_function_clusters)
            
            return has_heavy and has_stable and has_movable and not is_in_excluded_affordance and not is_in_excluded_function
        
        return False
    
    def _analyze_material_inference_questions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze material inference question types"""
        results = {}
        
        # Sound absorption questions (textiles, fabrics)
        textile_materials = ['Textiles, Fibers & Leather']
        sound_absorbing_objects_set = set()
        
        for obj_name in self.all_objects:
            clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            if any(cluster in clusters for cluster in textile_materials):
                sound_absorbing_objects_set.add(obj_name)
        
        sound_absorbing_objects = sorted(list(sound_absorbing_objects_set))
        
        results['material_sound_absorption'] = {
            'description': 'objects that best absorb sound (textiles and fabrics)',
            'filter_criteria': textile_materials,
            'total_objects_tested': len(self.all_objects),
            'valid_objects': sound_absorbing_objects,
            'valid_count': len(sound_absorbing_objects)
        }
        
        # Thermal touch questions (metals, stone, glass)
        thermal_materials = ['Metals & Alloys', 'Stone, Concrete & Mineral', 'Glass & Transparent (Silicate)']
        thermal_objects = []
        
        for obj_name in self.all_objects:
            clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            if any(cluster in clusters for cluster in thermal_materials):
                thermal_objects.append(obj_name)
        
        results['material_thermal_touch'] = {
            'description': 'objects that would feel cold to touch (metals, stone, glass)',
            'filter_criteria': thermal_materials,
            'total_objects_tested': len(self.all_objects),
            'valid_objects': sorted(thermal_objects),
            'valid_count': len(thermal_objects)
        }
        
        # Scratch resistance questions (hard materials)
        hard_materials = ['Metals & Alloys', 'Stone, Concrete & Mineral', 'Ceramics, Porcelain & Earthenware']
        scratch_resistant_objects = []
        
        for obj_name in self.all_objects:
            clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            if any(cluster in clusters for cluster in hard_materials):
                scratch_resistant_objects.append(obj_name)
        
        results['material_scratch_resistance'] = {
            'description': 'objects with surfaces least likely to scratch (hard materials)',
            'filter_criteria': hard_materials,
            'total_objects_tested': len(self.all_objects),
            'valid_objects': sorted(scratch_resistant_objects),
            'valid_count': len(scratch_resistant_objects)
        }
        
        return results
    
    def _analyze_description_based_questions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze description-based question types"""
        results = {}
        
        # Define void/abstract clusters that should be excluded from description matching
        void_clusters = {
            'No-Physical-Properties',
            'No clear affordance', 
            'Organic / Abstract / No Definite Shape (Animals/Humans/Roles/Concepts)',
            'Atextural/Symbolic (Documents/Icons/Light-Only Events)',
            'Unclassified'
        }
        
        def is_in_void_cluster(obj_name: str) -> bool:
            """Check if object is in any void/abstract cluster"""
            try:
                # Check all taxonomy types for void clusters
                taxonomy_types = [
                    'final_taxonomy_affordances',
                    'final_taxonomy_physical_properties', 
                    'final_taxonomy_materials',
                    'final_taxonomy_texture',
                    'final_taxonomy_shape'
                ]
                
                for tax_type in taxonomy_types:
                    clusters = self.taxonomy_utils.get_object_clusters(obj_name, tax_type)
                    if any(cluster in void_clusters for cluster in clusters):
                        return True
                return False
            except:
                return False
        
        # Load object descriptions to check which objects have descriptions
        descriptions_file = Path("/path/to/SpatialReasonerDataGen/object_description/results/object_list_final/full_object_descriptions_fully_parsed.json")
        
        objects_with_descriptions = []
        objects_with_material_descriptions = []
        objects_with_function_descriptions = []
        
        if descriptions_file.exists():
            try:
                with open(descriptions_file, 'r', encoding='utf-8') as f:
                    descriptions = json.load(f)
                
                for obj_name in self.all_objects:
                    # Skip objects in void clusters
                    if is_in_void_cluster(obj_name):
                        continue
                        
                    obj_descriptions = descriptions.get(obj_name, {})
                    
                    # Check for general description (handle both string and list formats)
                    general_desc = obj_descriptions.get('general_description', [])
                    if isinstance(general_desc, list):
                        has_general_desc = any(item and str(item).strip() for item in general_desc)
                    else:
                        has_general_desc = general_desc and str(general_desc).strip()
                    if has_general_desc:
                        objects_with_descriptions.append(obj_name)
                    
                    # Check for material description (handle both string and list formats)
                    material_desc = obj_descriptions.get('material', [])
                    if isinstance(material_desc, list):
                        has_material_desc = any(item and str(item).strip() for item in material_desc)
                    else:
                        has_material_desc = material_desc and str(material_desc).strip()
                    if has_material_desc:
                        objects_with_material_descriptions.append(obj_name)
                    
                    # Check for function description (handle both string and list formats)
                    func_desc = obj_descriptions.get('functions', [])
                    if isinstance(func_desc, list):
                        has_func_desc = any(item and str(item).strip() for item in func_desc)
                    else:
                        has_func_desc = func_desc and str(func_desc).strip()
                    if has_func_desc:
                        objects_with_function_descriptions.append(obj_name)
                        
            except Exception as e:
                print(f"Warning: Could not load object descriptions: {e}")
        
        
        results['material_property'] = self._create_result_entry(
            'objects with material descriptions for material property questions',
            ['objects with non-empty material attribute'],
            objects_with_material_descriptions
        )
        
        results['function_knowledge'] = self._create_result_entry(
            'objects with function descriptions for function knowledge questions',
            ['objects with non-empty functions attribute'],
            objects_with_function_descriptions
        )
        
        return results
    
    def _analyze_counterfactual_questions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze counterfactual question types"""
        results = {}
        
        # Water-sensitive objects
        water_sensitive_materials = [
            'Paper, Cardboard & Pulp', 
            'Biological (Plants/Flowers)', 
            'Organic Food & Edible Matter'
        ]
        water_sensitive_objects_set = set()
        
        for obj_name in self.all_objects:
            clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            if any(cluster in clusters for cluster in water_sensitive_materials):
                water_sensitive_objects_set.add(obj_name)
        
        water_sensitive_objects = sorted(list(water_sensitive_objects_set))
        
        results['counterfactual_water'] = {
            'description': 'objects that would be damaged by water spills',
            'filter_criteria': water_sensitive_materials,
            'total_objects_tested': len(self.all_objects),
            'valid_objects': water_sensitive_objects,
            'valid_count': len(water_sensitive_objects)
        }
        
        # Heat-sensitive objects
        heat_sensitive_materials = ['Textiles, Fibers & Leather', 'Plastics, Rubber & Polymers', 'Paper, Cardboard & Pulp']
        heat_sensitive_objects_set = set()
        
        for obj_name in self.all_objects:
            clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            if any(cluster in clusters for cluster in heat_sensitive_materials):
                heat_sensitive_objects_set.add(obj_name)
        
        heat_sensitive_objects = sorted(list(heat_sensitive_objects_set))
        
        results['counterfactual_heat'] = {
            'description': 'objects most affected by high heat',
            'filter_criteria': heat_sensitive_materials,
            'total_objects_tested': len(self.all_objects),
            'valid_objects': heat_sensitive_objects,
            'valid_count': len(heat_sensitive_objects)
        }
        
        return results
    
    def _analyze_latent_state_questions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze latent state question types"""
        results = {}
        
        # Containment objects (hollow or container affordance)
        containment_objects_set = set()
        
        # Check each object for container affordance or hollow property
        for obj_name in self.all_objects:
            # Check for container affordance
            obj_affordance_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_affordances')
            has_container_affordance = 'Contain / Carry / Package' in obj_affordance_clusters
            
            # Check for hollow property
            obj_physical_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_physical_properties')
            has_hollow_property = 'Hollow' in obj_physical_clusters
            
            # Require BOTH container affordance AND hollow property for stronger filtering
            has_both_container_criteria = has_container_affordance and has_hollow_property
            
            # EXCLUDE objects that are primarily equipment, infrastructure, or sports equipment using affordance clusters
            exclude_affordance_clusters = [
                'Operate / Use Device',  
                'Mechanical Control',  
                'Build / Span / Occupy',           
                'Sit / Ride / Attend',             
                'Control / Express / Light',       
                'Household / Facility Operations' 
            ]
            is_in_excluded_affordance = any(cluster in obj_affordance_clusters for cluster in exclude_affordance_clusters)
            
            # EXCLUDE function clusters that are typically equipment/infrastructure
            obj_function_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_function')
            exclude_function_clusters = [
                'Medical & Healthcare Equipment',
                'Devices, Displays, Transactions & Admin',
                'Architecture, Rooms & Built Spaces', 
                'Vehicles, Transportation ',
                'Sports & Recreation Equipment'
            ]
            is_in_excluded_function = any(cluster in obj_function_clusters for cluster in exclude_function_clusters)
            
            # EXCLUDE objects that are fragile (not suitable for hiding items)
            has_fragile = 'Fragile' in obj_physical_clusters
            
            # EXCLUDE objects that are unstable (not suitable for hiding items)
            has_unstable = 'Unstable' in obj_physical_clusters
            
            # Only include if it meets container criteria AND is not equipment/infrastructure
            if has_both_container_criteria and not is_in_excluded_affordance and not is_in_excluded_function and not has_fragile and not has_unstable:
                containment_objects_set.add(obj_name)
        
        containment_objects = sorted(list(containment_objects_set))
        
        results['latent_containment'] = {
            'description': 'objects that can hide items (hollow or container affordance)',
            'filter_criteria': [
                'REQUIRE: Contain / Carry / Package affordance AND Hollow physical property',
                'EXCLUDE: Operate / Use Device affordance',
                'EXCLUDE: Mechanical Control affordance',
                'EXCLUDE: Build / Span / Occupy affordance', 
                'EXCLUDE: Sit / Ride / Attend affordance',
                'EXCLUDE: Control / Express / Light affordance',
                'EXCLUDE: Household / Facility Operations affordance',
                'EXCLUDE: Medical & Healthcare Equipment function',
                'EXCLUDE: Devices, Displays, Transactions & Admin function',
                'EXCLUDE: Architecture, Rooms & Built Spaces function',
                'EXCLUDE: Vehicles, Transportation function',
                'EXCLUDE: Sports & Recreation Equipment function',
                'EXCLUDE: Fragile physical property',
                'EXCLUDE: Unstable physical property'
            ],
            'total_objects_tested': len(self.all_objects),
            'valid_objects': sorted(containment_objects),
            'valid_count': len(containment_objects)
        }
        
        # Compressible objects (textiles, soft materials)
        compressible_objects_set = set()
        
        # Use cluster-based filtering for compressible materials
        compressible_material_clusters = [
            'Textiles, Fibers & Leather',
            'Plastics, Rubber & Polymers'
        ]
        
        for obj_name in self.all_objects:
            # Check material clusters for compressible materials
            material_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_material')
            is_compressible_material = any(cluster in compressible_material_clusters for cluster in material_clusters)
            
            # Must also have flexible physical property
            has_flexible = self.taxonomy_utils.has_property(obj_name, 'flexible')
            
            # EXCLUDE rigid physical property - objects cannot be both flexible and rigid
            has_rigid = self.taxonomy_utils.has_property(obj_name, 'rigid')
            
            if is_compressible_material and has_flexible and not has_rigid:
                compressible_objects_set.add(obj_name)
        
        compressible_objects = sorted(list(compressible_objects_set))
        
        results['latent_compressible'] = {
            'description': 'objects that can be compressed to fit in tight spaces',
            'filter_criteria': compressible_material_clusters + ['Flexible physical property'],
            'total_objects_tested': len(self.all_objects),
            'valid_objects': compressible_objects,
            'valid_count': len(compressible_objects)
        }
        
        return results

    def _check_flatness_for_shield(self, obj_name: str) -> bool:
        """Check if object has proper flatness for shield use"""
        obj_lower = obj_name.lower()
        
        flat_shape_cluster_names = [
            "Square / Rectangle (Flat)",
            "Circle / Disk", 
            "Plate / Sheet / Thin Flat"
        ]
        
        # Get shape clusters for this object
        try:
            shape_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_shape')
            cluster_text = ' '.join(shape_clusters).lower()
            
            # Check if any of the flat shape cluster names appear in cluster_text
            for cluster_name in flat_shape_cluster_names:
                if cluster_name.lower() in cluster_text:
                    return True
        except:
            pass  # If we can't get clusters, continue with other checks
                
        # Additional flat shield objects not in taxonomy clusters
        additional_flat_objects = [
            'tray', 'pan', 'grill', 'briefcase', 'game board', 'paddle', 'weight plate',
            'baking sheet', 'frying pan', 'jigsaw puzzle', 'lunch box'
        ]
        
        if any(flat in obj_lower for flat in additional_flat_objects):
            return True
        
        
        non_flat_objects = [
            # Cylindrical containers
            'bottle', 'can', 'cylinder', 'jar', 'cup', 'jug', 'kitchen bottle', 'milk bottle', 
            'shot glass', 'spice jar', 'water bottle', 'flask', 'thermos', 'kettle', 'teapot',
            'mug', 'glass', 'tumbler', 'vase', 'bowl', 'dish',
            # Spherical/round objects
            'ball', 'cue ball', 'baseball', 'softball', 'bell',
            # Pots and containers
            'pot', 'cooking pot', 'coffee pot', 'bucket', 'canister', 'mop bucket', 'plastic bucket',
            'colander', 'shopping basket', 'trash can', 'trashcan',
            # Not flat structures
            'flowerpot', 'cupboard box', 'stand', 'holder', 'candle stand', 'chopstick stand',
            # Small utensils
            'fork', 'spoon', 'scissors', 'teaspoon', 'whisk', 'tong', 'stapler',
            # Writing implements
            'pen', 'pen holder', 'pen marker', 'pencil jar', 'pencil sharpener', 'marker',
            # Other small items
            'earring', 'ring', 'tiara', 'toy', 'barbie', 'clip'
        ]
        
        # If it's explicitly non-flat, exclude it
        if any(non_flat in obj_lower for non_flat in non_flat_objects):
            return False
            
        # If it's explicitly flat, include it
        if any(flat in obj_lower for flat in flat_shield_objects):
            return True
            
        
        try:
            shape_clusters = self.taxonomy_utils.get_object_clusters(obj_name, 'final_taxonomy_shape')
            cluster_text = ' '.join(shape_clusters).lower()
            
            flat_indicators = ['flat', 'plate', 'sheet', 'board', 'rectangular', 'square']
            curved_indicators = ['cylindrical', 'spherical', 'curved', 'round', 'circular']
            
            has_flat_indicators = any(indicator in cluster_text for indicator in flat_indicators)
            has_curved_indicators = any(indicator in cluster_text for indicator in curved_indicators)
            
            return has_flat_indicators and not has_curved_indicators
        except:
            
            return True
    
    def _check_shield_size(self, obj_name: str) -> bool:
        """Check if object is appropriate size for shield use"""
        obj_lower = obj_name.lower()
        
        # Too small to be effective shields
        small_objects = [
            'earring', 'ring', 'tiara', 'clip', 'pen', 'marker', 'spoon', 'fork',
            'chopstick', 'salt shaker', 'shaker', 'medal', 'plug hat', 'razor',
            'candle stand', 'chopstick stand', 'stand', 'holder', 'clip', 'pin',
            'button', 'badge', 'coin', 'token', 'key', 'keychain', 'watch', 'bracelet',
            'bell', 'scissors', 'teaspoon', 'whisk', 'tong', 'stapler', 'pen holder',
            'pen marker', 'pencil jar', 'pencil sharpener', 'shot glass', 'spice jar',
            'utensil', 'umbrella stub', 'stick', 'sticks', 'rulers'
        ]
        
        # Too large to be practical shields  
        large_objects = [
            'pallet', 'stage', 'football field', 'woodpallet', 'shipping boxes'
        ]
        
        # Additional keyword-based exclusions for shield concept
        shield_exclusion_keywords = [
            'bottle', 'bucket', 'can', 'pot', 'jar', 'cup', 'glass', 'mug', 'bowl', 'dish',
            'ball', 'bell', 'fork', 'spoon', 'scissors', 'pen', 'marker', 'clip', 'stand',
            'holder', 'basket', 'trash', 'utensil', 'stick', 'rulers', 'toy', 'prop'
        ]
        
        if any(small in obj_lower for small in small_objects):
            return False
        if any(large in obj_lower for large in large_objects):
            return False
            
        return True

    def save_results(self, results: Dict[str, Dict[str, Any]], output_file: str):
        """Save results to JSON file with separated SM and OpenImages objects"""
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to new format with separated objects
        converted_results = {}
        
        for q_type, data in results.items():
            # Check if already in new format
            if 'sm_valid_objects' in data:
                converted_results[q_type] = data
            else:
                
                valid_objects = data.get('valid_objects', [])
                separated = self._separate_objects_by_type(valid_objects)
                
                # Create new entry with separated SM and OpenImages objects only
                new_data = {k: v for k, v in data.items() if k not in ['valid_objects', 'valid_count', 'sm_valid_objects', 'openimages_valid_objects', 'sm_objects_count', 'openimages_objects_count', 'sm_percentage', 'openimages_percentage']}
                new_data.update({
                    'sm_valid_objects': separated['sm_objects'],
                    'openimages_valid_objects': separated['openimages_objects'],
                    'sm_objects_count': separated['sm_count'],
                    'openimages_objects_count': separated['openimages_count'],
                    'sm_percentage': round(separated['sm_count'] / len(valid_objects) * 100, 1) if valid_objects else 0.0,
                    'openimages_percentage': round(separated['openimages_count'] / len(valid_objects) * 100, 1) if valid_objects else 0.0
                })
                converted_results[q_type] = new_data
        
        output_data = {
            'metadata': {
                'total_objects_in_taxonomy': len(self.all_objects),
                'question_types_analyzed': len(converted_results)
            },
            'question_answer_mappings': converted_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")
        
        print(f"Total objects in taxonomy: {len(self.all_objects)}")
        print(f"Question types analyzed: {len(converted_results)}")
        
        for q_type, data in converted_results.items():
            sm_count = data.get('sm_objects_count', 0)
            openimages_count = data.get('openimages_objects_count', 0)
            total_valid = sm_count + openimages_count
            total_count = data.get('total_objects_tested', len(self.all_objects))
            percentage = (total_valid / total_count * 100) if total_count > 0 else 0
            print(f"  {q_type}: {total_valid}/{total_count} objects ({percentage:.1f}%) - {sm_count} SM, {openimages_count} OpenImages")


def main():
    parser = argparse.ArgumentParser(description='Analyze complete answer space for all question types')
    parser.add_argument('--taxonomy-dir', 
                       default='/path/to/SpatialReasonerDataGen/qa_gen/taxonomy',
                       help='Directory containing taxonomy data')
    parser.add_argument('--output-file', 
                       default=None,
                       help='Output JSON file path (default: scripts/analysis/results/question_answer_space_analysis.json relative to qa_gen directory)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided - resolve relative to qa_gen directory
    if args.output_file is None:
        script_dir = Path(__file__).parent.parent.parent
        args.output_file = str(script_dir / "scripts/analysis/results/question_answer_space_analysis.json")
    else:
        # If relative path provided, resolve relative to qa_gen directory
        output_path = Path(args.output_file)
        if not output_path.is_absolute():
            script_dir = Path(__file__).parent.parent.parent
            args.output_file = str(script_dir / output_path)
    
    
    analyzer = QuestionAnswerSpaceAnalyzer(args.taxonomy_dir)

    results = analyzer.analyze_all_question_types()
    
    analyzer.save_results(results, args.output_file)


if __name__ == "__main__":
    main()
