import json
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class TaxonomyUtils:
    def __init__(self, taxonomy_dir: Path):
        self.taxonomy_clusters = self.load_taxonomy_clusters(taxonomy_dir)
    
    def load_taxonomy_clusters(self, taxonomy_dir: Path) -> Dict[str, Dict[str, Any]]:
        clusters = {}
        
        if taxonomy_dir.exists():
            for taxonomy_file in taxonomy_dir.glob("*.json"):
                key = taxonomy_file.stem
                try:
                    with open(taxonomy_file, 'r') as f:
                        clusters[key] = json.load(f)
                    logger.info(f"Loaded {key} taxonomy")
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping corrupted taxonomy file {taxonomy_file.name}: {e}")
                    continue
        
        return clusters
    
    def get_object_clusters(self, class_name: str, taxonomy_type: str) -> List[str]:
        clusters_found = []
        class_name_lower = class_name.lower()
        
        taxonomy_data = self.taxonomy_clusters.get(taxonomy_type, {})
        
        if taxonomy_type == 'final_taxonomy_physical_properties':
            # For physical properties, use the new separate cluster structure
            key = 'physical_properties'
        else:
            key = taxonomy_type.replace('final_taxonomy_', '')
            # Fix key mapping for affordances (key is 'affordance', not 'affordances')
            if taxonomy_type == 'final_taxonomy_affordances' and 'affordance' in taxonomy_data:
                key = 'affordance'
        
        clusters = taxonomy_data.get(key, {})
        
        for cluster_name, cluster_data in clusters.items():
            cluster_objects = [obj.lower() for obj in cluster_data.get('objects', [])]
            if class_name_lower in cluster_objects:
                clusters_found.append(cluster_name)
        
        return clusters_found
    
    def get_objects_in_light_flexible_clusters(self, object_names: List[str]) -> List[str]:
        light_flexible_objects = []
        
        physical_props_taxonomy = self.taxonomy_clusters.get('final_taxonomy_physical_properties', {})
        clusters = physical_props_taxonomy.get('physical_properties', {})
        
        for cluster_name, cluster_data in clusters.items():
            if 'Light' in cluster_name and 'Flexible' in cluster_name:
                cluster_objects = [obj.lower() for obj in cluster_data.get('objects', [])]
                for obj_name in object_names:
                    if obj_name.lower() in cluster_objects and obj_name not in light_flexible_objects:
                        light_flexible_objects.append(obj_name)
        
        return light_flexible_objects
    
    def get_objects_with_smooth_texture(self, object_names: List[str]) -> List[str]:
        smooth_objects = []
        
        texture_taxonomy = self.taxonomy_clusters.get('final_taxonomy_texture', {})
        clusters = texture_taxonomy.get('texture', {})
        
        for cluster_name, cluster_data in clusters.items():
            if 'smooth' in cluster_name.lower():
                cluster_objects = [obj.lower() for obj in cluster_data.get('objects', [])]
                for obj_name in object_names:
                    if obj_name.lower() in cluster_objects and obj_name not in smooth_objects:
                        smooth_objects.append(obj_name)
        
        return smooth_objects
    
    def get_flammable_objects(self, object_names: List[str]) -> List[str]:
        flammable_objects = []
        flammable_materials = ['wood', 'fabric', 'organic', 'textile']
        
        material_taxonomy = self.taxonomy_clusters.get('final_taxonomy_material', {})
        clusters = material_taxonomy.get('material', {})
        
        for cluster_name, cluster_data in clusters.items():
            cluster_name_lower = cluster_name.lower()
            if any(mat in cluster_name_lower for mat in flammable_materials):
                cluster_objects = [obj.lower() for obj in cluster_data.get('objects', [])]
                for obj_name in object_names:
                    if obj_name.lower() in cluster_objects and obj_name not in flammable_objects:
                        flammable_objects.append(obj_name)
        
        return flammable_objects
    
    def has_property(self, class_name: str, property_name: str) -> bool:
        """
        Check if an object has a specific property using separate property clusters
        
        Args:
            class_name: Object class name
            property_name: Property to check (e.g., 'hollow', 'rigid', 'movable')
            
        Returns:
            True if object has the property
        """
        class_name_lower = class_name.lower()
        
        # Get the new separate property clusters structure
        taxonomy_data = self.taxonomy_clusters.get('final_taxonomy_physical_properties', {})
        property_clusters = taxonomy_data.get('physical_properties', {})
        
        property_title = property_name.title()
        if property_title not in property_clusters:
            return False
        
        cluster_data = property_clusters[property_title]
        cluster_objects = [obj.lower() for obj in cluster_data.get('objects', [])]
        
        return class_name_lower in cluster_objects

    