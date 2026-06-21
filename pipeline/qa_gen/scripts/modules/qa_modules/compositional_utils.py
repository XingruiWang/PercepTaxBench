#!/usr/bin/env python3
"""
Compositional Reasoning Utilities

Helper functions for compositional negation, attribute swaps, and role reassignment.
"""

import re
from typing import Dict, List, Tuple, Optional, Any


class CompositionalUtils:
    """Utilities for compositional reasoning questions"""
    
    def __init__(self, taxonomy_utils):
        self.taxonomy_utils = taxonomy_utils
    
    def check_properties(self, class_name: str, required_props: List[str], 
                        forbidden_props: List[str] = None) -> bool:
        """
        Check if object has all required properties and none of the forbidden ones
        Uses separate property clusters for precise filtering
        
        Args:
            class_name: Object class name
            required_props: List of required property keywords
            forbidden_props: List of forbidden property keywords
            
        Returns:
            True if object matches criteria
        """
        if forbidden_props is None:
            forbidden_props = []
        
        # First try to use separate property clusters for physical properties
        physical_properties = [
            'light', 'heavy', 'rigid', 'flexible', 'fragile', 'movable', 'fixed', 
            'stable', 'unstable', 'solid-core', 'hollow', 'conductive', 'insulating',
            'durable', 'smooth', 'rough', 'solid', 'liquid', 'gas'
        ]
        
        properties_checked = set(required_props + forbidden_props)
        can_use_separate_clusters = all(prop.lower() in physical_properties for prop in properties_checked)
        
        # Special handling for affordance-based forbidden properties before other checks
        if 'contain' in forbidden_props:
            affordance_clusters = self.taxonomy_utils.get_object_clusters(
                class_name, 'final_taxonomy_affordances'
            )
            # Check exact cluster name first
            if 'Contain / Carry / Package' in affordance_clusters:
                return False
            
            # Container keywords that indicate the object is designed to contain/store things
            container_keywords = ['container', 'bottle', 'cup', 'mug', 'jug', 'pot', 
                               'cabinet', 'crate', 'jar', 'can', 'box', 'bag', 'basket',
                               'barrel', 'bucket', 'tank', 'vessel', 'bin', 'drawer']
            
            # If object name contains container keywords, exclude it
            if any(keyword in class_name.lower() for keyword in container_keywords):
                return False
            
            # Also check related affordance clusters that indicate container functionality
            # combined with container-like keywords in object name (additional safety check)
            container_related_clusters = [
                'Tableware & Serveware',  # cups, mugs, jugs, pots
                'Household / Facility Operations',  # cabinets, storage
            ]
            if any(cluster in affordance_clusters for cluster in container_related_clusters):
                # If in container-related cluster and has container keyword, exclude (redundant but safe)
                if any(keyword in class_name.lower() for keyword in container_keywords):
                    return False
        
        if can_use_separate_clusters:
            # Use precise separate property clusters
            for prop in required_props:
                if not self.taxonomy_utils.has_property(class_name, prop):
                    return False
            
            for prop in forbidden_props:
                if self.taxonomy_utils.has_property(class_name, prop):
                    return False
            
            return True
        else:
            # Fallback to original method for non-physical properties or mixed properties
            all_clusters = []
            for tax_type in ['final_taxonomy_physical_properties', 'final_taxonomy_affordances', 
                            'final_taxonomy_material', 'final_taxonomy_texture']:
                clusters = self.taxonomy_utils.get_object_clusters(class_name, tax_type)
                all_clusters.extend(clusters)
            
            cluster_str = ' '.join(all_clusters).lower()
            
            # Use improved pattern matching to handle special characters in cluster names
            for prop in required_props:
                prop_lower = prop.lower()
                # Handle cases where property might be surrounded by special characters like / and ·
                pattern = r'(?:^|[^a-zA-Z0-9])' + re.escape(prop_lower) + r'(?:$|[^a-zA-Z0-9])'
                if not re.search(pattern, cluster_str):
                    return False
            
            for prop in forbidden_props:
                # Skip 'contain' since it was already handled above
                if prop.lower() == 'contain':
                    continue
                prop_lower = prop.lower()
                # Handle cases where property might be surrounded by special characters like / and ·
                pattern = r'(?:^|[^a-zA-Z0-9])' + re.escape(prop_lower) + r'(?:$|[^a-zA-Z0-9])'
                if re.search(pattern, cluster_str):
                    return False
            
            return True
    
    def get_material_type(self, class_name: str, object_descriptions: Dict) -> Optional[str]:
        """Get primary material type of object"""
        class_lower = class_name.lower()
        if class_lower not in object_descriptions:
            return None
        
        materials = object_descriptions[class_lower].get('material', [])
        if not materials:
            return None
        
        material_str = ' '.join(materials).lower()
        
        if any(m in material_str for m in ['plastic', 'polymer']):
            return 'plastic'
        elif any(m in material_str for m in ['metal', 'steel', 'aluminum', 'iron']):
            return 'metal'
        elif any(m in material_str for m in ['wood', 'timber', 'oak', 'pine']):
            return 'wood'
        elif any(m in material_str for m in ['glass', 'crystal']):
            return 'glass'
        elif any(m in material_str for m in ['fabric', 'textile', 'cloth', 'cotton', 'leather']):
            return 'fabric'
        elif any(m in material_str for m in ['paper', 'cardboard']):
            return 'paper'
        elif any(m in material_str for m in ['rubber', 'latex']):
            return 'rubber'
        
        return None
    
    def get_texture_type(self, class_name: str) -> List[str]:
        """Get texture types of object"""
        texture_clusters = self.taxonomy_utils.get_object_clusters(
            class_name, 'final_taxonomy_texture'
        )
        
        textures = []
        cluster_str = ' '.join(texture_clusters).lower()
        
        if 'smooth' in cluster_str:
            textures.append('smooth')
        if 'rough' in cluster_str:
            textures.append('rough')
        if 'soft' in cluster_str:
            textures.append('soft')
        if 'hard' in cluster_str:
            textures.append('hard')
        
        return textures
    
    def get_object_shape(self, class_name: str) -> List[str]:
        """Get shape descriptors of object"""
        phys_clusters = self.taxonomy_utils.get_object_clusters(
            class_name, 'final_taxonomy_physical_property'
        )
        
        shapes = []
        cluster_str = ' '.join(phys_clusters).lower()
        
        if 'flat' in cluster_str:
            shapes.append('flat')
        if 'hollow' in cluster_str:
            shapes.append('hollow')
        if 'solid' in cluster_str:
            shapes.append('solid')
        
        return shapes
    
    def has_affordance(self, class_name: str, affordance_keyword: str) -> bool:
        """Check if object has specific affordance using exact cluster matching"""
        aff_clusters = self.taxonomy_utils.get_object_clusters(
            class_name, 'final_taxonomy_affordances'
        )
        # Use exact cluster name matching instead of substring matching for better precision
        affordance_keyword_lower = affordance_keyword.lower()
        return any(cluster.lower() == affordance_keyword_lower or 
                  affordance_keyword_lower in cluster.lower() for cluster in aff_clusters)
    
    def filter_by_criteria(self, detected_objects: List[Dict], 
                          required: List[str], 
                          forbidden: List[str] = None,
                          min_count: int = 1) -> List[Tuple[str, str]]:
        """
        Filter objects by required and forbidden properties
        
        Returns:
            List of (labeled_name, class_name) tuples
        """
        if forbidden is None:
            forbidden = []
        
        matches = []
        for obj in detected_objects:
            class_name = obj['class_name'].lower()
            obj_name = obj.get('labeled_name', obj['class_name'])
            
            if self.check_properties(class_name, required, forbidden):
                matches.append((obj_name, class_name))
        
        return matches if len(matches) >= min_count else []
    
    def get_material_swap_properties(self, from_material: str, to_material: str) -> Dict[str, str]:
        """
        Get property changes when swapping materials
        
        Returns:
            Dict with 'property_change' and 'benefit_or_drawback'
        """
        swaps = {
            ('plastic', 'rubber'): {
                'property': 'grip and flexibility',
                'change': 'better grip (less slippery) and more flexible'
            },
            ('metal', 'wood'): {
                'property': 'weight',
                'change': 'lighter and easier to move'
            },
            ('wood', 'metal'): {
                'property': 'durability and heat resistance',
                'change': 'more durable and heat-resistant'
            },
            ('glass', 'plastic'): {
                'property': 'fragility',
                'change': 'less fragile and safer (reduced breakage risk)'
            },
            ('fabric', 'leather'): {
                'property': 'water resistance',
                'change': 'more water-resistant and durable'
            },
            ('paper', 'plastic'): {
                'property': 'water resistance',
                'change': 'more water-resistant (won\'t dissolve when wet)'
            },
        }
        
        return swaps.get((from_material, to_material), {
            'property': 'material properties',
            'change': 'different physical characteristics'
        })
    
    def get_repurposed_use(self, property_type: str) -> Optional[Dict[str, str]]:
        """
        Get repurposed use case for object property
        
        Returns:
            Dict with 'goal', 'action', and 'reason'
        """
        uses = {
            'soft_flexible': {
                'goal': 'make {furniture} more comfortable',
                'action': 'Use {object} as cushion/support on {furniture}',
                'reason': 'soft material provides comfort'
            },
            'flat_rigid': {
                'goal': 'make {small_object} more visible',
                'action': 'Place {small_object} on top of {object} to elevate it',
                'reason': 'flat surface provides stable platform'
            },
            'container': {
                'goal': 'reduce clutter from scattered items',
                'action': 'Store small items inside {object}',
                'reason': 'container provides organized storage'
            },
            'heavy': {
                'goal': 'prevent {light_object} from moving',
                'action': 'Place {object} on top of {light_object} to weigh it down',
                'reason': 'weight provides stability'
            },
            'reflective': {
                'goal': 'brighten a dark area',
                'action': 'Angle {object} to reflect light toward the dark area',
                'reason': 'reflective surface redirects light'
            },
        }
        
        return uses.get(property_type)

