#!/usr/bin/env python3

import json
import logging
import math
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from .data_loading_utils import DataLoadingUtils
from .filter_utils import HIGH_RELIABILITY_CLASSES, is_void_cluster

logger = logging.getLogger(__name__)


class ObjectUtils:
    """Utilities for object-related operations used in QA generation"""
    
    def __init__(self, taxonomy_utils=None):
        self.taxonomy_utils = taxonomy_utils
        self.data_loading_utils = DataLoadingUtils()
        self.object_descriptions = self.data_loading_utils.load_object_descriptions()
        self.name_mappings = self.data_loading_utils.load_name_mappings()
        # Cache annotation data to avoid repeated disk reads when generating reasoning
        self._annotation_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
    
    def get_object_description(self, object_name: str) -> str:
        """Get description for an object"""
        if not self.object_descriptions:
            return ""
        
        # Try to find the object in descriptions
        obj_data = self.object_descriptions.get(object_name)
        if not obj_data:
            return ""
        
        # Get description from the object data
        description = obj_data.get('description', '')
        if description:
            # Clean description in case it contains malformed text
            return self._clean_malformed_text(description)
        
        # If no direct description, try to build one from available fields
        description_parts = []
        
        # Add material if available
        material = obj_data.get('material', [])
        if material and isinstance(material, list) and len(material) > 0:
            if isinstance(material[0], str):
                material_text = self._clean_malformed_text(material[0])
                description_parts.append(f"made of {material_text}")
        
        # Add function if available
        functions = obj_data.get('functions', [])
        if functions and isinstance(functions, list) and len(functions) > 0:
            if isinstance(functions[0], str):
                function_text = self._clean_malformed_text(functions[0])
                description_parts.append(f"used for {function_text}")
        
        # Return first non-empty description snippet
        if description_parts:
            return description_parts[0]
        
        return ""
    
    def _clean_malformed_text(self, text: str) -> str:
        """Clean malformed text like unclosed parentheses, extra quotes, etc."""
        if not text:
            return ""
        
        cleaned = text.strip()
        
        # Remove common introductory phrases that make descriptions awkward
        intro_phrases = [
            "Commonly made from",
            "Commonly constructed from",
            "Commonly built from",
            "Commonly manufactured from",
            "Typically made from",
            "Typically constructed from",
            "Usually made from"
        ]
        
        for phrase in intro_phrases:
            if cleaned.startswith(phrase):
                cleaned = cleaned[len(phrase):].strip()
        
        # Handle unclosed parentheses - split on '(', keep only the part before
        if '(' in cleaned:
            # Count parentheses to see if it's balanced
            open_count = cleaned.count('(')
            close_count = cleaned.count(')')
            if open_count > close_count:
                # Unclosed parentheses detected - take everything before the last '('
                parts = cleaned.rsplit('(', 1)
                cleaned = parts[0].strip() if parts[0] else cleaned
        
        # Remove trailing incomplete quotes
        if cleaned and cleaned[-1] in ['\'', '"']:
            cleaned = cleaned[:-1].strip()
        
        # Remove leading incomplete quotes
        if cleaned and cleaned[0] in ['\'', '"']:
            cleaned = cleaned[1:].strip()
        
        # Remove any stray single quotes at the start or end
        cleaned = cleaned.strip("'\"")
        
        return cleaned
    
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

    # ------------------------------------------------------------------
    # Annotation helpers for spatial reasoning
    # ------------------------------------------------------------------

    def _load_annotations_for_image(
        self,
        image_id: str,
        annotations_dir: Optional[Path],
    ) -> Optional[Dict[str, Any]]:
        """Load annotation JSON for a given image with caching."""
        if not annotations_dir or not image_id:
            return None

        cache_key = (str(annotations_dir), image_id)
        if cache_key in self._annotation_cache:
            return self._annotation_cache[cache_key]

        candidate_files: List[Path] = []

        # Primary layout (OpenImages-style): <annotations_dir>/<image_id>/annotations/<files>.json
        image_dir = annotations_dir / image_id / "annotations"
        if image_dir.exists():
            candidate_files.extend([
                image_dir / f"{image_id}_refined.json",
                image_dir / f"{image_id}.json",
            ])
            if not candidate_files[0].exists() and not candidate_files[1].exists():
                candidate_files.extend(sorted(image_dir.glob("*.json")))
        else:
            # Fallback layout used by simulation metadata:
            # <annotations_dir>/<scene>/<view>/scene_annotations_split.json
            alt_dir = annotations_dir / image_id
            if alt_dir.exists():
                candidate_files.extend([
                    alt_dir / "scene_annotations_split.json",
                    alt_dir / f"{Path(image_id).name}.json",
                ])
                candidate_files.extend(sorted((alt_dir / "annotations").glob("*.json"))) if (alt_dir / "annotations").exists() else None

        if not candidate_files:
            logger.debug("Annotation directory missing for %s under %s", image_id, annotations_dir)
            self._annotation_cache[cache_key] = None
            return None

        annotations_data: Optional[Dict[str, Any]] = None
        for candidate in candidate_files:
            if candidate.exists():
                try:
                    annotations_data = json.loads(candidate.read_text())
                    break
                except Exception as exc:
                    logger.warning("Failed to load annotation file %s: %s", candidate, exc)

        if annotations_data is None:
            logger.debug("No usable annotation file found for %s in %s", image_id, image_dir)

        self._annotation_cache[cache_key] = annotations_data
        return annotations_data

    @staticmethod
    def _normalize_object_name_for_lookup(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return str(name).strip().lower()

    def _index_detections_by_name(self, detections: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create a lookup from normalized object name to detection entry."""
        index: Dict[str, Dict[str, Any]] = {}
        for detection in detections or []:
            possible_names: List[str] = []
            for key in ("class_name", "class", "object_name"):
                value = detection.get(key)
                if isinstance(value, str):
                    # For object_name like "obj_00_market stall", strip prefixes
                    norm = value.split("_", 2)[-1] if key == "object_name" and "_" in value else value
                    possible_names.append(norm)
            for name in possible_names:
                normalized = self._normalize_object_name_for_lookup(name)
                if normalized and normalized not in index:
                    index[normalized] = detection
        return index

    def _parse_xyxy(self, xyxy: Union[str, List[float], Tuple[float, ...]]) -> Optional[List[float]]:
        """Parse an XYXY bbox representation into a list of floats."""
        if isinstance(xyxy, str):
            try:
                cleaned = xyxy.strip("[]").replace(",", " ").strip()
                values = [float(x) for x in cleaned.split() if x.strip()]
                return values if len(values) == 4 else None
            except Exception:
                return None
        if isinstance(xyxy, (list, tuple)):
            try:
                values = [float(x) for x in xyxy]
                return values if len(values) == 4 else None
            except Exception:
                return None
        return None

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        x_min, y_min, x_max, y_max = bbox
        return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

    def _extract_yaw_degrees(self, detection: Dict[str, Any]) -> Optional[float]:
        """Extract yaw orientation in degrees from detection metadata."""
        orientation_sources = [
            detection.get("pcd_orient_bbox"),
            detection.get("pcd_cano_orient_bbox"),
            detection.get("pcd_axis_bbox"),
        ]
        for source in orientation_sources:
            if isinstance(source, dict):
                eulers = source.get("eulers")
                vector = self._parse_vector(eulers)
                if vector is not None and len(vector) == 3:
                    # Assume third component corresponds to yaw around vertical axis
                    yaw_rad = float(vector[2])
                    yaw_deg = math.degrees(yaw_rad)
                    # Normalize yaw to [-180, 180] for readability
                    if yaw_deg > 180.0:
                        yaw_deg -= 360.0
                    if yaw_deg < -180.0:
                        yaw_deg += 360.0
                    return yaw_deg
        return None

    def get_spatial_reasoning_details(
        self,
        image_id: Optional[str],
        question_type: str,
        target_object: Optional[str],
        reference_object: Optional[str],
        annotations_dir: Optional[Path],
        target_label: Optional[str] = None,
        reference_label: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed spatial metrics (bbox centers, orientation, deltas) for reasoning.
        """
        if not image_id or not target_object or not reference_object:
            return None

        annotations = self._load_annotations_for_image(image_id, annotations_dir)
        if not annotations:
            return None

        detections = annotations.get("detections") or []
        if (not detections) and isinstance(annotations.get("foreground"), dict):
            sim_detections: List[Dict[str, Any]] = []
            foreground = annotations.get("foreground") or {}
            for object_name, entries in foreground.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    det: Dict[str, Any] = {
                        "class_name": object_name,
                        "object_name": object_name,
                        "xyxy": entry.get("bbox_2d"),
                    }
                    rotation = (
                        entry.get("bbox_3d", {}).get("rotation")
                        or entry.get("bbox_3d", {}).get("obb", {}).get("rotation")
                    )
                    if rotation and isinstance(rotation, (list, tuple)) and len(rotation) == 3:
                        try:
                            det["pcd_orient_bbox"] = {
                                "eulers": [math.radians(float(angle)) for angle in rotation]
                            }
                        except Exception:
                            pass
                    sim_detections.append(det)
            if sim_detections:
                detections = sim_detections

        detection_index = self._index_detections_by_name(detections)

        target_det = detection_index.get(self._normalize_object_name_for_lookup(target_object))
        reference_det = detection_index.get(self._normalize_object_name_for_lookup(reference_object))

        if not target_det or not reference_det:
            return None

        target_bbox = self._parse_xyxy(target_det.get("xyxy") or target_det.get("bbox"))
        reference_bbox = self._parse_xyxy(reference_det.get("xyxy") or reference_det.get("bbox"))

        if not target_bbox or not reference_bbox:
            return None

        target_center = self._bbox_center(target_bbox)
        reference_center = self._bbox_center(reference_bbox)

        target_yaw = self._extract_yaw_degrees(target_det)
        reference_yaw = self._extract_yaw_degrees(reference_det)

        spatial_relationship = self.get_spatial_relationship(
            target_object,
            reference_object,
            image_id,
            annotations_dir=annotations_dir,
        )

        relation_value = spatial_relationship.get(
            {
                "spatial_left_right": "left_right",
                "spatial_front_behind": "front_behind",
                "spatial_above_below": "above_below",
                "spatial_closer_to_camera": "closer",
            }.get(question_type, ""),
            "unknown",
        )

        dx = target_center[0] - reference_center[0]
        dy = target_center[1] - reference_center[1]
        distance = math.hypot(dx, dy)

        axis_description = ""
        if question_type == "spatial_left_right":
            if math.isclose(dx, 0.0, abs_tol=1e-3):
                axis_description = "The horizontal centers are aligned."
            else:
                direction = "right" if dx > 0 else "left"
                axis_description = (
                    f"The horizontal center offset is {abs(dx):.1f}px toward the {direction}."
                )
        elif question_type == "spatial_above_below":
            if math.isclose(dy, 0.0, abs_tol=1e-3):
                axis_description = "The vertical centers are aligned."
            else:
                direction = "below" if dy > 0 else "above"
                axis_description = (
                    f"The vertical center offset is {abs(dy):.1f}px toward the {direction}."
                )
        elif question_type == "spatial_front_behind":
            if math.isclose(dy, 0.0, abs_tol=1e-3):
                axis_description = "The depth proxy offset is negligible."
            else:
                orientation = (
                    "front" if relation_value == "front"
                    else "behind" if relation_value == "behind"
                    else "undetermined depth"
                )
                axis_description = (
                    f"The depth proxy offset is {abs(dy):.1f}px, indicating {orientation}."
                )
        elif question_type == "spatial_closer_to_camera":
            if math.isclose(dy, 0.0, abs_tol=1e-3):
                axis_description = "Vertical ordering suggests both objects are at similar depth."
            else:
                closer_hint = (
                    "the target appears lower in the frame (closer to camera)"
                    if dy < 0 else
                    "the reference appears lower in the frame (closer to camera)"
                )
                axis_description = (
                    f"The vertical ordering offset is {abs(dy):.1f}px, so {closer_hint}."
                )

        return {
            "target": {
                "object": target_object,
                "label": target_label,
                "center": target_center,
                "bbox": target_bbox,
                "yaw_deg": target_yaw,
            },
            "reference": {
                "object": reference_object,
                "label": reference_label,
                "center": reference_center,
                "bbox": reference_bbox,
                "yaw_deg": reference_yaw,
            },
            "delta": {
                "dx": dx,
                "dy": dy,
                "distance": distance,
                "axis_description": axis_description,
            },
            "spatial_relationship": spatial_relationship,
            "relation_value": relation_value,
        }
    
    def get_object_function(self, object_name: str) -> str:
        """Get function for an object, with cleaning and validation"""
        if not self.taxonomy_utils:
            # Try object descriptions first if no taxonomy utils
            return self._get_function_from_descriptions(object_name)
        
        try:
            function_clusters = self.taxonomy_utils.get_object_clusters(object_name, 'final_taxonomy_function')
            if function_clusters:
                # Check void clusters first
                from modules.qa_modules.filter_utils import is_void_cluster
                non_void_clusters = [cluster for cluster in function_clusters if not is_void_cluster(cluster, 'function')]
                
                # If object is only in void clusters, return empty (hard rejection - exclude from function questions)
                if not non_void_clusters:
                    return ""
                
                # Try to clean and validate each non-void cluster
                for cluster in non_void_clusters:
                    cleaned = self._clean_and_validate_function(cluster)
                    if cleaned:
                        return cleaned
        except Exception as e:
            logger.warning(f"Could not get function from taxonomy for {object_name}: {e}")
        
        # Fallback to object descriptions
        return self._get_function_from_descriptions(object_name)
        
    
    def _get_function_from_descriptions(self, object_name: str) -> str:
        """Get function from object descriptions"""
        try:
            obj_key = object_name.lower()
            if obj_key in self.object_descriptions:
                obj_data = self.object_descriptions[obj_key]
                functions = obj_data.get('functions', [])
                if functions:
                    if isinstance(functions, list) and len(functions) > 0:
                        # Return first non-empty function
                        for func in functions:
                            if func and isinstance(func, str):
                                func_stripped = func.strip()
                                # Skip overly long strings that look like descriptions
                                if len(func_stripped) > 100:
                                    continue
                                if "The description for" in func_stripped or "under the semantic key" in func_stripped:
                                    continue
                                # Filter out void cluster names (affordance, function, etc.)
                                if self._is_void_cluster_name(func_stripped):
                                    continue
                                if len(func_stripped) > 2:
                                    clean_func = self._clean_malformed_text(func_stripped)
                                    if clean_func and not self._is_void_cluster_name(clean_func):
                                        return clean_func
                    elif isinstance(functions, str) and functions.strip():
                        func_stripped = functions.strip()
                        # Skip overly long strings that look like descriptions
                        if len(func_stripped) > 100:
                            return ""
                        if "The description for" in func_stripped or "under the semantic key" in func_stripped:
                            return ""
                        # Filter out void cluster names
                        if self._is_void_cluster_name(func_stripped):
                            return ""
                        if len(func_stripped) > 2:
                            clean_func = self._clean_malformed_text(func_stripped)
                            if clean_func and not self._is_void_cluster_name(clean_func):
                                return clean_func
        except Exception as e:
            logger.warning(f"Could not get function from object descriptions for {object_name}: {e}")
        
        return ""
    
    def get_object_physical_properties(self, object_name: str) -> List[str]:
        """Get cleaned list of physical property cluster names for an object."""
        properties: List[str] = []
        
        if self.taxonomy_utils:
            try:
                clusters = self.taxonomy_utils.get_object_clusters(object_name, 'final_taxonomy_physical_properties')
                if clusters:
                    for cluster in clusters:
                        if not cluster or is_void_cluster(cluster, 'physical'):
                            continue
                        cleaned = self._clean_and_validate_physical_property(cluster)
                        if cleaned and cleaned not in properties:
                            properties.append(cleaned)
            except Exception as e:
                logger.warning(f"Could not get physical properties from taxonomy for {object_name}: {e}")
        
        # Fallback to object descriptions if taxonomy lookup failed
        if not properties:
            try:
                obj_key = object_name.lower()
                if obj_key in self.object_descriptions:
                    obj_data = self.object_descriptions[obj_key]
                    physical_props = obj_data.get('physical_properties', [])
                    if isinstance(physical_props, list):
                        for prop in physical_props:
                            if isinstance(prop, str):
                                cleaned = self._clean_and_validate_physical_property(prop)
                                if cleaned and cleaned not in properties:
                                    properties.append(cleaned)
                    elif isinstance(physical_props, str):
                        cleaned = self._clean_and_validate_physical_property(physical_props)
                        if cleaned:
                            properties.append(cleaned)
            except Exception as e:
                logger.warning(f"Could not get physical properties from object descriptions for {object_name}: {e}")
        
        return properties
    
    def _is_void_cluster_name(self, text: str) -> bool:
        """Check if text is a void cluster name that should not be used as function/material"""
        if not text or not isinstance(text, str):
            return False
        
        text_lower = text.lower().strip()
        
        # Check against known void cluster names (exact matches from taxonomy files)
        void_cluster_names = [
            'no clear function',  # Function void cluster
            'no clear affordance',  # Affordance void cluster (if it appears)
            'unclassified',
            'roles, occupation, and directed actions',
            'abstract / depictions / scenes/ occupations',
            'biological (animals/body parts)',
            'composites & multi‑material products',
            'no-physical-properties',
            'human roles & identities (occupations/person types)',
            'natural scenes (view/appraise)',
            'phenomena (view/read/appraise)',
            'atextural/symbolic (documents/icons/light-only events)',
            'organic / abstract / no definite shape (animals/humans/roles/concepts)'
        ]
        
        # Exact match check
        if text_lower in void_cluster_names:
            return True
        
        # Partial match check (in case of variations)
        for void_name in void_cluster_names:
            if void_name in text_lower or text_lower in void_name:
                return True
        
        return False
    
    def _clean_and_validate_material(self, material: str) -> str:
        """Clean and validate material string, return empty string if invalid"""
        if not material or not isinstance(material, str):
            return ""
        
        material = material.strip()
        
        # Reject void cluster materials based on content
        void_indicators = [
            'biological (animals/body parts)', 'biological tissue', 'biological tissues',
            'human body', 'flesh', 'composed of flesh', 'composed of human',
            'biological (human body)', 'biological entities', 'biological entity of',
            'the organic materials composing'
        ]
        material_lower = material.lower()
        if any(indicator in material_lower for indicator in void_indicators):
            return ""
        
        # Reject overly long strings (>40 chars) - likely descriptions, not material names
        if len(material) > 40:
            return ""
        
        # Reject sentence-like patterns
        sentence_patterns = [
            'commonly made', 'constructed from', 'primarily constructed', 
            'include', 'composed primarily', 'most frequently', 'materials for',
            'the primary materials', 'the main frame', 'not applicable',
            'the description contains', 'the organic materials'
        ]
        if any(pattern in material_lower for pattern in sentence_patterns):
            return ""
        
        # Reject references to object IDs (e.g., "Object 1", "Object 2")
        if re.search(r'\bObject\s+\d+', material, re.IGNORECASE):
            return ""
        
        # Clean up material string
        material = self._clean_malformed_text(material)
        material = material.replace('_', ' ')
        material = re.sub(r'\s+', ' ', material).strip()
        
        # Normalize common prefixes
        prefixes_to_remove = [
            'Primarily ', 'primarily ', 'Composed of ', 'composed of ',
            'Constructed from ', 'constructed from ', 'Made of ', 'made of '
        ]
        for prefix in prefixes_to_remove:
            if material.startswith(prefix):
                material = material[len(prefix):].strip()
        
        # Final validation: must have reasonable length after cleaning
        if len(material) < 2 or len(material) > 40:
            return ""
        
        return material
    
    def _clean_and_validate_function(self, function: str) -> str:
        """Clean and validate function cluster name, return empty string if invalid"""
        if not function or not isinstance(function, str):
            return ""
        
        function = function.strip()
        
        # Reject void cluster functions (should already be filtered, but double-check)
        from modules.qa_modules.filter_utils import is_void_cluster
        if is_void_cluster(function, 'function'):
            return ""
        
        # Reject overly long strings (>100 chars) - likely descriptions, not cluster names
        if len(function) > 100:
            return ""
        
        # Reject sentence-like patterns that indicate descriptions rather than cluster names
        sentence_patterns = [
            'commonly used', 'typically used', 'primarily used', 'used for',
            'the primary', 'the main', 'not applicable', 'the description',
            'to live', 'to work', 'to create'  # Description-style patterns
        ]
        function_lower = function.lower()
        if any(pattern in function_lower for pattern in sentence_patterns):
            return ""
        
        # Reject references to object IDs (e.g., "Object 1", "Object 2")
        if re.search(r'\bObject\s+\d+', function, re.IGNORECASE):
            return ""
        
        # Clean up function string
        function = self._clean_malformed_text(function)
        function = function.replace('_', ' ')
        function = re.sub(r'\s+', ' ', function).strip()
        
        # Normalize common prefixes (unlikely for cluster names, but safe)
        prefixes_to_remove = [
            'Primarily ', 'primarily ', 'Used for ', 'used for ',
            'Designed for ', 'designed for ', 'Intended for ', 'intended for '
        ]
        for prefix in prefixes_to_remove:
            if function.startswith(prefix):
                function = function[len(prefix):].strip()
        
        # Final validation: must have reasonable length after cleaning
        # Function cluster names can be longer than materials (up to ~50 chars)
        if len(function) < 2 or len(function) > 100:
            return ""
        
        return function
    
    def _clean_and_validate_physical_property(self, prop: str) -> str:
        """Clean and normalize physical property cluster names"""
        if not prop or not isinstance(prop, str):
            return ""
        
        cleaned = self._clean_malformed_text(prop)
        cleaned = cleaned.strip()
        
        if not cleaned:
            return ""
        
        # Normalize separators for readability
        cleaned = cleaned.replace('—', ' ').replace('_', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Reject overly long strings (likely descriptions)
        if len(cleaned) > 60:
            return ""
        
        # Lowercase for natural language usage
        cleaned = cleaned.strip().lower()
        
        return cleaned
    
    def get_object_material(self, object_name: str) -> str:
        """Get material for an object, with cleaning and validation"""
        if not self.taxonomy_utils:
            return ""
        
        try:
            material_clusters = self.taxonomy_utils.get_object_clusters(object_name, 'final_taxonomy_material')
            if material_clusters:
                # Check void clusters first - if ALL clusters are void, return empty (don't fall back to descriptions)
                from modules.qa_modules.filter_utils import is_void_cluster
                non_void_clusters = [cluster for cluster in material_clusters if not is_void_cluster(cluster, 'material')]
                
                # If object is only in void clusters, return empty (exclude from material questions)
                if not non_void_clusters:
                    return ""
                
                # Only use non-void clusters
                for cluster in non_void_clusters:
                    # Clean and validate the material
                    cleaned = self._clean_and_validate_material(cluster)
                    if cleaned:
                        return cleaned
        except Exception as e:
            logger.warning(f"Could not get material from taxonomy for {object_name}: {e}")
        
        # Fallback: try to get from object descriptions with proper validation
        # But first check if object would be in void cluster via affordance (person/occupation exclusion)
        try:
            # Check if this is a person/occupation (should be excluded from material questions)
            affordance_clusters = self.taxonomy_utils.get_object_clusters(object_name, 'final_taxonomy_affordances')
            if affordance_clusters and 'Human Roles & Identities (Occupations/Person Types)' in affordance_clusters:
                return ""  # Exclude person/occupation objects from material fallback too
            
            obj_key = object_name.lower()
            if obj_key in self.object_descriptions:
                obj_data = self.object_descriptions[obj_key]
                materials = obj_data.get('material', [])
                if materials:
                    material_list = materials if isinstance(materials, list) else [materials]
                    for material in material_list:
                        cleaned = self._clean_and_validate_material(material)
                        if cleaned:
                            return cleaned
        except Exception as e:
            logger.warning(f"Could not get material from object descriptions for {object_name}: {e}")
        
        return ""
    
    def get_spatial_relationship(self, object1: str, object2: str, image_id: str, 
                               annotations_dir: Path = None, 
                               object_poses: Dict = None,
                               depth_map_path: Path = None) -> Dict[str, str]:
        """Get spatial relationship between two objects - handles both real and sim images"""
        if object_poses:
            # Sim image: use pose data (and optionally depth map) directly
            if depth_map_path:
                return self._calculate_from_poses_with_depth(object1, object2, object_poses, depth_map_path)
            return self._calculate_from_poses(object1, object2, object_poses)
        else:
            # Real image: use annotation files
            return self._calculate_from_annotations(object1, object2, image_id, annotations_dir)
    def _calculate_from_poses(self, object1: str, object2: str, object_poses: Dict) -> Dict[str, str]:
        """Calculate spatial relationship from sim image pose data using hybrid approach"""
        try:
            pose1 = object_poses.get(object1, {})
            pose2 = object_poses.get(object2, {})
            
            if not pose1 or not pose2:
                logger.warning(f"Missing pose data for {object1} or {object2}")
                return {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown"}
            
            # Get 3D locations from pose data
            location1 = pose1.get('location', [0, 0, 0])
            location2 = pose2.get('location', [0, 0, 0])
            
            if len(location1) != 3 or len(location2) != 3:
                logger.warning(f"Invalid pose coordinates: {location1}, {location2}")
                return {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown"}
            
            # Calculate 3D distance for overall significance check
            from modules.qa_modules.spatial_utils import SpatialUtils
            distance_3d = SpatialUtils.calculate_3d_distance(location1, location2)
            
            # If objects are too close overall, relationships are ambiguous
            min_distance_3d = 0.08
            if distance_3d < min_distance_3d:
                logger.debug(f"Objects {object1} and {object2} too close (distance={distance_3d:.3f} < {min_distance_3d}), returning unknown relationships")
                return {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown"}
            
            # Prepare object data for hybrid calculation (compatible with real image format)
            obj1_data = pose1.copy()
            obj2_data = pose2.copy()
            
            # Use location as pcd_center for compatibility with hybrid methods
            obj1_data['pcd_center'] = location1
            obj2_data['pcd_center'] = location2
            
            # Get 2D bboxes if available
            bbox1_2d = pose1.get('bbox_2d', [])
            bbox2_2d = pose2.get('bbox_2d', [])
            
            # LEFT/RIGHT: Use hybrid approach (geometric orientation if available, otherwise fallback to 3D X comparison)
            left_right = "unknown"
            if len(bbox1_2d) == 4 and len(bbox2_2d) == 4:
                # Use hybrid approach with bbox overlap checks and geometric orientation
                left_right = self._calculate_left_right_hybrid(obj1_data, obj2_data, bbox1_2d, bbox2_2d)
                logger.debug(f"Left/right (hybrid) for sim: {object1} vs {object2}, result={left_right}")
            
            # Fallback to 3D X coordinate comparison if bbox not available or hybrid returned unknown
            if left_right == "unknown":
                x1, y1, z1 = location1
                x2, y2, z2 = location2
                x_diff = abs(x2 - x1)
                min_separation = 0.05
                if x_diff >= min_separation:
                    if x1 < x2:
                        left_right = "left"  # obj1 is left of obj2
                    elif x1 > x2:
                        left_right = "right"  # obj1 is right of obj2
            
            # ABOVE/BELOW: Use Z-axis (vertical) comparison
            x1, y1, z1 = location1
            x2, y2, z2 = location2
            z_diff = abs(z2 - z1)
            min_separation = 0.05
            
            above_below = "unknown"
            
            # Check if both objects are on the ground (Z coordinate close to 0 or very low)
            # Objects on the ground should have Z coordinates near ground level
            ground_level_threshold = 0.15  # 15cm - objects below this are considered on ground
            both_on_ground = z1 < ground_level_threshold and z2 < ground_level_threshold
            
            if both_on_ground:
                # Both objects are on the ground - mark as unknown (they're at the same level)
                above_below = "unknown"
                logger.debug(f"Above/below: both objects on ground (z1={z1:.3f}, z2={z2:.3f}), marking as unknown")
            elif z_diff >= min_separation:
                if z1 > z2:
                    above_below = "above"  # obj1 is above obj2
                elif z1 < z2:
                    above_below = "below"  # obj1 is below obj2
            
            # FRONT/BEHIND: Use hybrid approach if orientation vectors available, otherwise Y-axis
            front_behind = "unknown"
            if len(bbox1_2d) == 4 and len(bbox2_2d) == 4:
                # Try geometric orientation first
                front_behind = self._calculate_front_behind_hybrid(obj1_data, obj2_data, bbox1_2d, bbox2_2d)
                logger.debug(f"Front/behind (hybrid) for sim: {object1} vs {object2}, result={front_behind}")
            
            # Fallback to Y-axis comparison if hybrid returned unknown
            if front_behind == "unknown":
                y_diff = abs(y2 - y1)
                if y_diff >= min_separation:
                    if y1 < y2:
                        front_behind = "front"  # obj1 is in front of obj2
                    elif y1 > y2:
                        front_behind = "behind"  # obj1 is behind obj2
            
            return {
                "left_right": left_right,
                "above_below": above_below,
                "front_behind": front_behind
            }
            
        except Exception as e:
            logger.error(f"Error calculating spatial relationship from poses for {object1} and {object2}: {e}")
            return {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown"}
    
    def _calculate_from_poses_with_depth(self, object1: str, object2: str, object_poses: Dict, depth_map_path: Path) -> Dict[str, str]:
        """Calculate spatial relationship using depth map for more accurate camera-perspective relationships"""
        try:
            import numpy as np
            
            # Load depth map
            if not depth_map_path.exists():
                logger.warning(f"Depth map not found: {depth_map_path}, using 3D coords")
                return self._calculate_from_poses(object1, object2, object_poses)
            
            depth_map = np.load(depth_map_path)
            
            # Get basic spatial relationship from 3D coordinates
            basic_spatial = self._calculate_from_poses(object1, object2, object_poses)
            
            # Get pose data with bounding boxes
            pose1 = object_poses.get(object1, {})
            pose2 = object_poses.get(object2, {})
            
            if pose1 and pose2:
                # Get bounding boxes for each object (if available)
                bbox1 = pose1.get('bbox_2d', None)
                bbox2 = pose2.get('bbox_2d', None)
                
                # Calculate depths using bounding boxes if available
                depth1 = None
                depth2 = None
                
                if bbox1:
                    x_min, y_min, x_max, y_max = bbox1
                    x_min = max(0, min(int(x_min), depth_map.shape[1] - 1))
                    y_min = max(0, min(int(y_min), depth_map.shape[0] - 1))
                    x_max = max(0, min(int(x_max), depth_map.shape[1] - 1))
                    y_max = max(0, min(int(y_max), depth_map.shape[0] - 1))
                    bbox_depth1 = depth_map[y_min:y_max+1, x_min:x_max+1]
                    if bbox_depth1.size > 0:
                        depth1 = np.mean(bbox_depth1[bbox_depth1 > 0])
                
                if bbox2:
                    x_min, y_min, x_max, y_max = bbox2
                    x_min = max(0, min(int(x_min), depth_map.shape[1] - 1))
                    y_min = max(0, min(int(y_min), depth_map.shape[0] - 1))
                    x_max = max(0, min(int(x_max), depth_map.shape[1] - 1))
                    y_max = max(0, min(int(y_max), depth_map.shape[0] - 1))
                    bbox_depth2 = depth_map[y_min:y_max+1, x_min:x_max+1]
                    if bbox_depth2.size > 0:
                        depth2 = np.mean(bbox_depth2[bbox_depth2 > 0])
                
                # If we have depth values for both objects, use them for front/behind
                if depth1 is not None and depth2 is not None:
                    if depth1 < depth2:  # obj1 is closer (smaller depth value)
                        basic_spatial["front_behind"] = "front"
                    elif depth1 > depth2:  # obj1 is farther
                        basic_spatial["front_behind"] = "behind"
                    else:
                        basic_spatial["front_behind"] = "unknown"
                    logger.debug(f"Depth-based spatial: {object1} ({depth1:.2f}) vs {object2} ({depth2:.2f})")
                
                # Fallback to Y-coordinate if no bbox depth available
                elif not bbox1 or not bbox2:
                    location1 = pose1.get('location', [0, 0, 0])
                    location2 = pose2.get('location', [0, 0, 0])
                    y1, y2 = location1[1], location2[1]
                    
                    if y1 < y2:
                        basic_spatial["front_behind"] = "front"
                    elif y1 > y2:
                        basic_spatial["front_behind"] = "behind"
                    else:
                        basic_spatial["front_behind"] = "unknown"
                    logger.debug(f"Y-coord fallback spatial: {object1} ({y1:.2f}) vs {object2} ({y2:.2f})")
                
                return basic_spatial
            
            return basic_spatial
            
        except Exception as e:
            logger.warning(f"Could not calculate spatial relationship with depth: {e}")
            # Fallback to basic calculation
            return self._calculate_from_poses(object1, object2, object_poses)
    
    def _parse_bbox_string(self, value):
        """Parse a bbox coordinate string like '[x y z]' into a list of floats"""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                cleaned = value.strip('[]').strip()
                parsed = [float(x) for x in cleaned.split() if x.strip()]
                if len(parsed) == 3:
                    return parsed
            except Exception as e:
                logger.debug(f"Could not parse bbox string '{value}': {e}")
        return None
    
    def _get_bbox_bounds_from_detection(self, detection: Dict) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Extract 3D bounding box bounds (min/max for each axis) from detection data.
        
        Args:
            detection: Detection dict with bbox data
            
        Returns:
            Dict with keys 'x', 'y', 'z', each containing (min, max) tuple.
            Returns None if bbox data is not available or invalid.
        """
        # Prefer canonical axis-aligned bbox (simpler, no rotation)
        bbox_data = detection.get('pcd_cano_axis_bbox')
        if not bbox_data:
            # Fall back to regular axis-aligned bbox
            bbox_data = detection.get('pcd_axis_bbox')
        
        if not bbox_data or not isinstance(bbox_data, dict):
            return None
        
        center_str = bbox_data.get('center')
        extent_str = bbox_data.get('extent')
        
        if not center_str or not extent_str:
            return None
        
        center = self._parse_bbox_string(center_str)
        extent = self._parse_bbox_string(extent_str)
        
        if center is None or extent is None or len(center) != 3 or len(extent) != 3:
            return None
        
        # For axis-aligned bbox: min = center - extent/2, max = center + extent/2
        half_extent = [e / 2.0 for e in extent]
        min_vals = [c - h for c, h in zip(center, half_extent)]
        max_vals = [c + h for c, h in zip(center, half_extent)]
        
        return {
            'x': (min_vals[0], max_vals[0]),
            'y': (min_vals[1], max_vals[1]),
            'z': (min_vals[2], max_vals[2])
        }
    
    def _calculate_spatial_from_bbox_bounds(self, bounds1: Dict[str, Tuple[float, float]], 
                                           bounds2: Dict[str, Tuple[float, float]]) -> Dict[str, str]:
        """
        Calculate spatial relationships using 3D bounding box boundaries.
        
        Args:
            bounds1: Bbox bounds for object1 {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            bounds2: Bbox bounds for object2 {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            
        Returns:
            Dict with 'left_right', 'above_below', 'front_behind' relationships
        """
        # Extract bounds
        x1_min, x1_max = bounds1['x']
        y1_min, y1_max = bounds1['y']
        z1_min, z1_max = bounds1['z']
        
        x2_min, x2_max = bounds2['x']
        y2_min, y2_max = bounds2['y']
        z2_min, z2_max = bounds2['z']
        
        # Calculate box sizes for relative threshold determination
        x1_size = x1_max - x1_min
        y1_size = y1_max - y1_min
        z1_size = z1_max - z1_min
        
        x2_size = x2_max - x2_min
        y2_size = y2_max - y2_min
        z2_size = z2_max - z2_min
        
        # Minimum separation thresholds: 30% of larger box dimension, at least 0.05 units
        min_x_separation = max(max(x1_size, x2_size) * 0.3, 0.05)
        min_y_separation = max(max(y1_size, y2_size) * 0.3, 0.05)
        min_z_separation = max(max(z1_size, z2_size) * 0.3, 0.05)
        
        # Left/Right (X-axis): obj1 is left if obj1's right edge is left of obj2's left edge
        left_right = "unknown"
        if x1_max < x2_min:
            # obj1 is completely to the left of obj2
            x_separation = x2_min - x1_max
            if x_separation >= min_x_separation:
                left_right = "left"
        elif x1_min > x2_max:
            # obj1 is completely to the right of obj2
            x_separation = x1_min - x2_max
            if x_separation >= min_x_separation:
                left_right = "right"
        # If boxes overlap in X or separation is too small, relationship is unknown
        
        # Above/Below (Z-axis): obj1 is above if obj1's bottom is above obj2's top
        above_below = "unknown"
        if z1_min > z2_max:
            # obj1 is completely above obj2
            z_separation = z1_min - z2_max
            if z_separation >= min_z_separation:
                above_below = "above"
        elif z1_max < z2_min:
            # obj1 is completely below obj2
            z_separation = z2_min - z1_max
            if z_separation >= min_z_separation:
                above_below = "below"
        # If boxes overlap in Z or separation is too small, relationship is unknown
        
        # Front/Behind (Y-axis): obj1 is in front if obj1's back is in front of obj2's front
        # (assuming negative Y = front, positive Y = behind)
        front_behind = "unknown"
        if y1_max < y2_min:
            # obj1 is completely in front of obj2 (obj1's max Y < obj2's min Y)
            y_separation = y2_min - y1_max
            if y_separation >= min_y_separation:
                front_behind = "front"
        elif y1_min > y2_max:
            # obj1 is completely behind obj2 (obj1's min Y > obj2's max Y)
            y_separation = y1_min - y2_max
            if y_separation >= min_y_separation:
                front_behind = "behind"
        # If boxes overlap in Y or separation is too small, relationship is unknown
        
        return {
            "left_right": left_right,
            "above_below": above_below,
            "front_behind": front_behind
        }
    
    def _is_high_reliability_orientation(self, class_name: str) -> bool:
        """Check if object class typically has reliable orientation (front/back/left/right)"""
        if not class_name:
            return False
        return class_name.lower() in HIGH_RELIABILITY_CLASSES
    
    def _parse_vector(self, vector_data: Any) -> Optional[np.ndarray]:
        """Parse vector from string, list, or array format"""
        if vector_data is None:
            return None
        
        try:
            if isinstance(vector_data, str):
                cleaned = vector_data.strip('[]').strip()
                vector = np.array([float(x) for x in cleaned.split() if x.strip()])
            elif isinstance(vector_data, list):
                vector = np.array([float(x) for x in vector_data])
            elif isinstance(vector_data, np.ndarray):
                vector = vector_data
            else:
                return None
            
            if len(vector) == 3:
                return vector
            return None
        except Exception:
            return None
    
    def _validate_orientation_vectors(self, front: np.ndarray, left: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Validate that orientation vectors are mathematically consistent"""
        if front is None or left is None:
            return False, "missing vectors"
        
        if len(front) != 3 or len(left) != 3:
            return False, "invalid vector length"
        
        # Check normalization
        front_norm = np.linalg.norm(front)
        left_norm = np.linalg.norm(left)
        if abs(front_norm - 1.0) > 0.01 or abs(left_norm - 1.0) > 0.01:
            return False, f"vectors not normalized (front={front_norm:.3f}, left={left_norm:.3f})"
        
        # Check orthogonality
        dot = np.dot(front, left)
        if abs(dot) > 0.1:
            return False, f"vectors not orthogonal (dot={dot:.3f})"
        
        return True, None
    
    def _get_reference_object(self, obj1_data: Dict, obj2_data: Dict) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
        """
        Identify which object is high-reliability and has pose data (reference object)
        
        Returns:
            (reference_data, target_data, which_is_reference) where which_is_reference is 'obj1', 'obj2', or None
        """
        class1 = obj1_data.get('class_name', '')
        class2 = obj2_data.get('class_name', '')
        
        obj1_is_high = self._is_high_reliability_orientation(class1)
        obj2_is_high = self._is_high_reliability_orientation(class2)
        
        # Check if object1 has pose data
        obj1_has_pose = ('left' in obj1_data and 'pcd_center' in obj1_data)
        
        # Check if object2 has pose data
        obj2_has_pose = ('left' in obj2_data and 'pcd_center' in obj2_data)
        
        # Prefer object2 as reference (standard approach), but use object1 if object2 doesn't qualify
        if obj2_is_high and obj2_has_pose:
            return obj2_data, obj1_data, 'obj2'
        elif obj1_is_high and obj1_has_pose:
            return obj1_data, obj2_data, 'obj1'
        else:
            return None, None, None
    
    def _check_separation_aspect_ratio(self, bbox1_2d: list, bbox2_2d: list) -> Tuple[float, float]:
        """
        Calculate separation aspect ratio to determine if objects are primarily
        vertically or horizontally separated.
        
        Returns:
            (x_separation, y_separation) - absolute separations in pixels
        """
        if len(bbox1_2d) != 4 or len(bbox2_2d) != 4:
            return (0.0, 0.0)
        
        # Calculate center positions
        cx1 = (bbox1_2d[0] + bbox1_2d[2]) / 2
        cx2 = (bbox2_2d[0] + bbox2_2d[2]) / 2
        cy1 = (bbox1_2d[1] + bbox1_2d[3]) / 2
        cy2 = (bbox2_2d[1] + bbox2_2d[3]) / 2
        
        x_separation = abs(cx2 - cx1)
        y_separation = abs(cy2 - cy1)
        
        return (x_separation, y_separation)
    
    def _check_horizontal_alignment(self, bbox1_2d: list, bbox2_2d: list, overlap_threshold: float = 0.3) -> bool:
        """
        Check if objects are horizontally aligned (significant overlap in both X and Y)
        
        Returns:
            True if objects are aligned (should skip question), False otherwise
        """
        if len(bbox1_2d) != 4 or len(bbox2_2d) != 4:
            return False
        
        x1_min, x1_max = bbox1_2d[0], bbox1_2d[2]
        x2_min, x2_max = bbox2_2d[0], bbox2_2d[2]
        y1_min, y1_max = bbox1_2d[1], bbox1_2d[3]
        y2_min, y2_max = bbox2_2d[1], bbox2_2d[3]
        
        # Check if bboxes overlap
        x_overlap = not (x1_max < x2_min or x1_min > x2_max)
        y_overlap = not (y1_max < y2_min or y1_min > y2_max)
        
        if x_overlap and y_overlap:
            # Calculate overlap ratios
            x_overlap_size = min(x1_max, x2_max) - max(x1_min, x2_min)
            obj1_width = x1_max - x1_min
            obj2_width = x2_max - x2_min
            x_overlap_ratio = max(x_overlap_size / obj1_width if obj1_width > 0 else 0,
                                 x_overlap_size / obj2_width if obj2_width > 0 else 0)
            
            y_overlap_size = min(y1_max, y2_max) - max(y1_min, y2_min)
            obj1_height = y1_max - y1_min
            obj2_height = y2_max - y2_min
            y_overlap_ratio = max(y_overlap_size / obj1_height if obj1_height > 0 else 0,
                                 y_overlap_size / obj2_height if obj2_height > 0 else 0)
            
            # If both X and Y overlap significantly, objects are aligned
            if x_overlap_ratio > overlap_threshold and y_overlap_ratio > overlap_threshold:
                return True
        
        return False
    
    def _calculate_left_right_from_orientation(self, reference_data: Dict, target_data: Dict) -> Optional[str]:
        """
        Calculate left/right using reference object's orientation (left vector)
        
        Returns: "left", "right", or None if calculation cannot be performed
        """
        # Get reference object's left vector
        left_vec = self._parse_vector(reference_data.get('left', []))
        if left_vec is None:
            return None
        
        # Get 3D centers
        reference_center = self._parse_vector(reference_data.get('pcd_center', []))
        target_center = self._parse_vector(target_data.get('pcd_center', []))
        
        if reference_center is None or target_center is None:
            return None
        
        # Validate vectors
        front_vec = self._parse_vector(reference_data.get('front', []))
        if front_vec is not None:
            is_valid, _ = self._validate_orientation_vectors(front_vec, left_vec)
            if not is_valid:
                return None
        
        # Calculate relative position
        relative_pos = target_center - reference_center
        
        # Project onto left vector
        dot_product = np.dot(relative_pos, left_vec)
        
        # Threshold for determination (0.1m = 10cm)
        threshold = 0.1
        
        if dot_product > threshold:
            return "left"  # target is to the left of reference (geometrically)
        elif dot_product < -threshold:
            return "right"  # target is to the right of reference (geometrically)
        else:
            return "unknown"  # Too close to determine
    
    def _calculate_front_behind_from_orientation(self, reference_data: Dict, target_data: Dict) -> Optional[str]:
        """
        Calculate front/behind using reference object's orientation (front vector)
        
        Returns: "front", "behind", or None if calculation cannot be performed
        """
        # Get reference object's front vector
        front_vec = self._parse_vector(reference_data.get('front', []))
        if front_vec is None:
            return None
        
        # Get 3D centers
        reference_center = self._parse_vector(reference_data.get('pcd_center', []))
        target_center = self._parse_vector(target_data.get('pcd_center', []))
        
        if reference_center is None or target_center is None:
            return None
        
        # Validate vectors
        left_vec = self._parse_vector(reference_data.get('left', []))
        if left_vec is not None:
            is_valid, _ = self._validate_orientation_vectors(front_vec, left_vec)
            if not is_valid:
                return None
        
        # Calculate relative position
        relative_pos = target_center - reference_center
        
        # Project onto front vector
        dot_product = np.dot(relative_pos, front_vec)
        
        # Threshold for determination (0.1m = 10cm)
        threshold = 0.1
        
        if dot_product > threshold:
            return "front"  # target is in front of reference (geometrically)
        elif dot_product < -threshold:
            return "behind"  # target is behind reference (geometrically)
        else:
            return "unknown"  # Too close to determine (similar depth)
    
    def _calculate_left_right_hybrid(self, obj1_data: Dict, obj2_data: Dict, bbox1_2d: list, bbox2_2d: list) -> str:
        """
        Calculate left/right using hybrid approach:
        - At least one object must be high-reliability
        - Check for horizontal alignment first
        - Check separation aspect ratio to skip if objects are primarily vertically separated
        - Use geometric orientation if available, otherwise fallback to viewer perspective
        
        Returns: "left", "right", or "unknown"
        """
        # Check if at least one object is high-reliability
        class1 = obj1_data.get('class_name', '')
        class2 = obj2_data.get('class_name', '')
        
        if not (self._is_high_reliability_orientation(class1) or self._is_high_reliability_orientation(class2)):
            # Both are low-reliability, skip geometric calculation
            # Use viewer perspective fallback (will be handled by caller)
            return "unknown"
        
        # Check for horizontal alignment
        if self._check_horizontal_alignment(bbox1_2d, bbox2_2d):
            return "unknown"
        
        # Check separation aspect ratio: skip left/right if objects are primarily vertically separated
        # If Y separation >> X separation, objects are above/below, not left/right
        x_sep, y_sep = self._check_separation_aspect_ratio(bbox1_2d, bbox2_2d)
        if x_sep > 0 and y_sep > 0:
            aspect_ratio = y_sep / x_sep if x_sep > 0 else float('inf')
            # If vertical separation is more than 2x horizontal separation, skip left/right
            if aspect_ratio > 2.0:
                logger.debug(f"Skipping left/right: objects are primarily vertically separated (Y_sep={y_sep:.1f}, X_sep={x_sep:.1f}, ratio={aspect_ratio:.2f})")
                return "unknown"
        
        # Try to use geometric orientation
        reference_data, target_data, which_ref = self._get_reference_object(obj1_data, obj2_data)
        
        if reference_data is not None and target_data is not None:
            result = self._calculate_left_right_from_orientation(reference_data, target_data)
            if result is not None:
                return result
        
        # Fallback to viewer perspective (2D bbox center X positions)
        if len(bbox1_2d) == 4 and len(bbox2_2d) == 4:
            cx1 = (bbox1_2d[0] + bbox1_2d[2]) / 2
            cx2 = (bbox2_2d[0] + bbox2_2d[2]) / 2
            cx_diff = abs(cx2 - cx1)
            min_2d_x_separation = 20.0
            
            if cx_diff >= min_2d_x_separation:
                if cx1 < cx2:
                    return "left"
                elif cx1 > cx2:
                    return "right"
        
        return "unknown"
    
    def _calculate_front_behind_hybrid(self, obj1_data: Dict, obj2_data: Dict, bbox1_2d: list, bbox2_2d: list) -> str:
        """
        Calculate front/behind using hybrid approach:
        - At least one object must be high-reliability
        - Use geometric orientation if available, otherwise fallback to 3D depth or 2D Y proxy
        
        Returns: "front", "behind", or "unknown"
        """
        # Check if at least one object is high-reliability
        class1 = obj1_data.get('class_name', '')
        class2 = obj2_data.get('class_name', '')
        
        if not (self._is_high_reliability_orientation(class1) or self._is_high_reliability_orientation(class2)):
            # Both are low-reliability, skip geometric calculation
            # Use 3D depth or 2D Y proxy fallback (will be handled by caller)
            return "unknown"
        
        # Try to use geometric orientation
        reference_data, target_data, which_ref = self._get_reference_object(obj1_data, obj2_data)
        
        if reference_data is not None and target_data is not None:
            result = self._calculate_front_behind_from_orientation(reference_data, target_data)
            if result is not None:
                return result
        
        # Fallback to 3D depth (Y-axis) or 2D Y proxy
        center1_3d = obj1_data.get('pcd_center', obj1_data.get('3d_data', {}).get('center', []))
        center2_3d = obj2_data.get('pcd_center', obj2_data.get('3d_data', {}).get('center', []))
        
        # Parse 3D centers if they're strings
        if isinstance(center1_3d, str):
            center1_3d = self._parse_vector(center1_3d)
        if isinstance(center2_3d, str):
            center2_3d = self._parse_vector(center2_3d)
        
        # Use 3D depth if available
        if isinstance(center1_3d, list) and isinstance(center2_3d, list) and len(center1_3d) == 3 and len(center2_3d) == 3:
            y1_3d, y2_3d = center1_3d[1], center2_3d[1]
            y_diff_3d = abs(y2_3d - y1_3d)
            min_3d_depth_separation = 0.1
            
            if y_diff_3d >= min_3d_depth_separation:
                # In world coordinates: smaller Y = closer to camera
                if y1_3d < y2_3d:
                    return "front"
                elif y1_3d > y2_3d:
                    return "behind"
        
        # Fallback to 2D Y proxy
        if len(bbox1_2d) == 4 and len(bbox2_2d) == 4:
            cy1 = (bbox1_2d[1] + bbox1_2d[3]) / 2
            cy2 = (bbox2_2d[1] + bbox2_2d[3]) / 2
            cy_diff = abs(cy2 - cy1)
            min_2d_y_separation = 10.0
            
            if cy_diff >= min_2d_y_separation:
                if cy1 > cy2:  # obj1 is lower in image (closer to camera)
                    return "front"
                elif cy1 < cy2:  # obj2 is lower in image (closer to camera)
                    return "behind"
        
        return "unknown"
    
    def _calculate_from_annotations(self, object1: str, object2: str, image_id: str, 
                                   annotations_dir: Path) -> Dict[str, str]:
        """Calculate spatial relationship from real image annotation files"""
        try:
            if not isinstance(image_id, str):
                logger.error("Invalid image_id type: %s (value=%s)", type(image_id), image_id)
                return {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown"}

            annotations = self._load_annotations_for_image(image_id, annotations_dir)
            if not annotations:
                logger.warning("No annotation data found for %s", image_id)
                return {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown"}

            detections = annotations.get("detections") or []
            if (not detections) and isinstance(annotations.get("foreground"), dict):
                sim_detections: List[Dict[str, Any]] = []
                foreground = annotations.get("foreground") or {}
                for object_name, entries in foreground.items():
                    if not isinstance(entries, list):
                        continue
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        det: Dict[str, Any] = {
                            "class_name": object_name,
                            "object_name": object_name,
                            "xyxy": entry.get("bbox_2d"),
                        }
                        rotation = (
                            entry.get("bbox_3d", {}).get("rotation")
                            or entry.get("bbox_3d", {}).get("obb", {}).get("rotation")
                        )
                        if rotation and isinstance(rotation, (list, tuple)) and len(rotation) == 3:
                            try:
                                det["pcd_orient_bbox"] = {
                                    "eulers": [math.radians(float(angle)) for angle in rotation]
                                }
                            except Exception:
                                pass
                        sim_detections.append(det)
                if sim_detections:
                    detections = sim_detections
            detection_index = self._index_detections_by_name(detections)

            obj1_data = detection_index.get(self._normalize_object_name_for_lookup(object1))
            obj2_data = detection_index.get(self._normalize_object_name_for_lookup(object2))

            if not obj1_data or not obj2_data:
                logger.warning("Objects not found in annotations for %s: %s, %s", image_id, object1, object2)
                return {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown"}
            
            # Get 2D bbox for spatial calculations
            bbox1_2d = obj1_data.get('bbox', obj1_data.get('xyxy', []))
            bbox2_2d = obj2_data.get('bbox', obj2_data.get('xyxy', []))
            
            # Parse bbox if it's a string (common format: "[ x1 y1 x2 y2 ]")
            if isinstance(bbox1_2d, str):
                try:
                    cleaned = bbox1_2d.strip('[]').strip()
                    bbox1_2d = [float(x) for x in cleaned.split() if x.strip()]
                except Exception as e:
                    logger.warning(f"Failed to parse bbox1_2d '{bbox1_2d}': {e}")
                    bbox1_2d = []
            
            if isinstance(bbox2_2d, str):
                try:
                    cleaned = bbox2_2d.strip('[]').strip()
                    bbox2_2d = [float(x) for x in cleaned.split() if x.strip()]
                except Exception as e:
                    logger.warning(f"Failed to parse bbox2_2d '{bbox2_2d}': {e}")
                    bbox2_2d = []
            
            spatial_result = {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown", "closer": "unknown"}
            
            if len(bbox1_2d) == 4 and len(bbox2_2d) == 4:
                cx1 = (bbox1_2d[0] + bbox1_2d[2]) / 2
                cy1 = (bbox1_2d[1] + bbox1_2d[3]) / 2
                cx2 = (bbox2_2d[0] + bbox2_2d[2]) / 2
                cy2 = (bbox2_2d[1] + bbox2_2d[3]) / 2

                # LEFT/RIGHT: Use hybrid approach (geometric orientation if available, otherwise viewer perspective)
                spatial_result["left_right"] = self._calculate_left_right_hybrid(obj1_data, obj2_data, bbox1_2d, bbox2_2d)
                logger.debug(f"Left/right (hybrid): {object1} vs {object2}, result={spatial_result['left_right']}")
                
                # Extract 2D bbox coordinates for above/below calculation
                y1_min_2d, y1_max_2d = bbox1_2d[1], bbox1_2d[3]
                y2_min_2d, y2_max_2d = bbox2_2d[1], bbox2_2d[3]
                
                # Check separation aspect ratio: skip above/below if objects are primarily horizontally separated
                # If X separation >> Y separation, objects are left/right, not above/below
                x_sep, y_sep = self._check_separation_aspect_ratio(bbox1_2d, bbox2_2d)
                if x_sep > 0 and y_sep > 0:
                    aspect_ratio = x_sep / y_sep if y_sep > 0 else float('inf')
                    # If horizontal separation is more than 2x vertical separation, skip above/below
                    if aspect_ratio > 2.0:
                        spatial_result["above_below"] = "unknown"
                        logger.debug(f"Skipping above/below: objects are primarily horizontally separated (X_sep={x_sep:.1f}, Y_sep={y_sep:.1f}, ratio={aspect_ratio:.2f})")
                    else:
                        # Calculate heights for relative comparison
                        h1 = y1_max_2d - y1_min_2d
                        h2 = y2_max_2d - y2_min_2d
                        avg_height = (h1 + h2) / 2
                        
                        # Check vertical overlap to detect objects on same surface
                        y_overlap_size = max(0, min(y1_max_2d, y2_max_2d) - max(y1_min_2d, y2_min_2d))
                        y_overlap_ratio = y_overlap_size / min(h1, h2) if min(h1, h2) > 0 else 0
                        
                        # Use bottom Y coordinates for above/below - objects on same surface should have similar bottom Y
                        # Bottom Y is more reliable than center Y because objects on a table will have similar bottom positions
                        bottom1 = y1_max_2d
                        bottom2 = y2_max_2d
                        bottom_diff = abs(bottom2 - bottom1)
                        
                        # Threshold: use percentage of average object height (at least 15% of object height)
                        # This prevents objects on the same surface from being marked as above/below
                        min_separation_threshold = max(avg_height * 0.15, 20.0)  # At least 15% of avg height or 20px
                        
                        # If objects overlap significantly vertically (>30%), they're likely on the same surface
                        if y_overlap_ratio > 0.3:
                            # Objects overlap significantly - likely on same surface, mark as unknown
                            spatial_result["above_below"] = "unknown"
                            logger.debug(f"Above/below ambiguous: significant vertical overlap (overlap_ratio={y_overlap_ratio:.2f}) - likely on same surface")
                        elif bottom_diff >= min_separation_threshold:
                            # Bottom positions are separated enough - determine above/below
                            if bottom1 < bottom2:  # obj1's bottom is higher (obj1 is above)
                                spatial_result["above_below"] = "above"
                            else:  # obj1's bottom is lower (obj1 is below)
                                spatial_result["above_below"] = "below"
                            logger.debug(f"Above/below from bottom Y: {object1} bottom={bottom1:.1f}, {object2} bottom={bottom2:.1f}, diff={bottom_diff:.1f}, result={spatial_result['above_below']}")
                        else:
                            # Bottom positions are too close - mark as unknown (likely on same surface)
                            spatial_result["above_below"] = "unknown"
                            logger.debug(f"Above/below ambiguous: bottom positions too close (diff={bottom_diff:.1f} < threshold={min_separation_threshold:.1f})")
                else:
                    # Fallback: if separation calculation failed, use original logic
                    # Calculate heights for relative comparison
                    h1 = y1_max_2d - y1_min_2d
                    h2 = y2_max_2d - y2_min_2d
                    avg_height = (h1 + h2) / 2
                    
                    # Check vertical overlap to detect objects on same surface
                    y_overlap_size = max(0, min(y1_max_2d, y2_max_2d) - max(y1_min_2d, y2_min_2d))
                    y_overlap_ratio = y_overlap_size / min(h1, h2) if min(h1, h2) > 0 else 0
                    
                    # Use bottom Y coordinates for above/below - objects on same surface should have similar bottom Y
                    # Bottom Y is more reliable than center Y because objects on a table will have similar bottom positions
                    bottom1 = y1_max_2d
                    bottom2 = y2_max_2d
                    bottom_diff = abs(bottom2 - bottom1)
                    
                    # Threshold: use percentage of average object height (at least 15% of object height)
                    # This prevents objects on the same surface from being marked as above/below
                    min_separation_threshold = max(avg_height * 0.15, 20.0)  # At least 15% of avg height or 20px
                    
                    # If objects overlap significantly vertically (>30%), they're likely on the same surface
                    if y_overlap_ratio > 0.3:
                        # Objects overlap significantly - likely on same surface, mark as unknown
                        spatial_result["above_below"] = "unknown"
                        logger.debug(f"Above/below ambiguous: significant vertical overlap (overlap_ratio={y_overlap_ratio:.2f}) - likely on same surface")
                    elif bottom_diff >= min_separation_threshold:
                        # Bottom positions are separated enough - determine above/below
                        if bottom1 < bottom2:  # obj1's bottom is higher (obj1 is above)
                            spatial_result["above_below"] = "above"
                        else:  # obj1's bottom is lower (obj1 is below)
                            spatial_result["above_below"] = "below"
                        logger.debug(f"Above/below from bottom Y: {object1} bottom={bottom1:.1f}, {object2} bottom={bottom2:.1f}, diff={bottom_diff:.1f}, result={spatial_result['above_below']}")
                    else:
                        # Bottom positions are too close - mark as unknown (likely on same surface)
                        spatial_result["above_below"] = "unknown"
                        logger.debug(f"Above/below ambiguous: bottom positions too close (diff={bottom_diff:.1f} < threshold={min_separation_threshold:.1f})")
                
                # FRONT/BEHIND: Use hybrid approach (geometric orientation if available, otherwise 3D depth or 2D Y proxy)
                spatial_result["front_behind"] = self._calculate_front_behind_hybrid(obj1_data, obj2_data, bbox1_2d, bbox2_2d)
                logger.debug(f"Front/behind (hybrid): {object1} vs {object2}, result={spatial_result['front_behind']}")
                
                # CLOSER TO CAMERA: Use 3D depth information when available, otherwise use 2D Y proxy
                # Try to get 3D center positions for depth calculation
                center1_3d_closer = obj1_data.get('pcd_center', obj1_data.get('3d_data', {}).get('center', []))
                center2_3d_closer = obj2_data.get('pcd_center', obj2_data.get('3d_data', {}).get('center', []))
                
                # Parse 3D centers if they're strings
                if isinstance(center1_3d_closer, str):
                    try:
                        cleaned = center1_3d_closer.strip('[]').strip()
                        center1_3d_closer = [float(x) for x in cleaned.split() if x.strip()]
                    except Exception:
                        center1_3d_closer = []
                
                if isinstance(center2_3d_closer, str):
                    try:
                        cleaned = center2_3d_closer.strip('[]').strip()
                        center2_3d_closer = [float(x) for x in cleaned.split() if x.strip()]
                    except Exception:
                        center2_3d_closer = []
                
                # Use 3D depth if available (Y-axis is depth in world coordinates)
                if isinstance(center1_3d_closer, list) and isinstance(center2_3d_closer, list) and len(center1_3d_closer) == 3 and len(center2_3d_closer) == 3:
                    y1_3d_closer, y2_3d_closer = center1_3d_closer[1], center2_3d_closer[1]
                    y_diff_3d_closer = abs(y2_3d_closer - y1_3d_closer)
                    min_3d_depth_separation_closer = 0.1  # 10cm minimum depth separation
                    
                    if y_diff_3d_closer >= min_3d_depth_separation_closer:
                        # In world coordinates: smaller Y = closer to camera
                        if y1_3d_closer < y2_3d_closer:
                            spatial_result["closer"] = object1  # object1 is closer
                        elif y1_3d_closer > y2_3d_closer:
                            spatial_result["closer"] = object2  # object2 is closer
                        logger.debug(f"Closer to camera from 3D depth: {object1} y={y1_3d_closer:.3f}, {object2} y={y2_3d_closer:.3f}, result={spatial_result['closer']}")
                    else:
                        # Depth difference too small, use 2D Y proxy as fallback
                        cy_diff_closer = abs(cy2 - cy1)
                        min_2d_y_separation_closer = 10.0
                        
                        if cy_diff_closer >= min_2d_y_separation_closer:
                            if cy1 > cy2:  # obj1 is lower in image (closer to camera)
                                spatial_result["closer"] = object1
                            elif cy1 < cy2:  # obj2 is lower in image (closer to camera)
                                spatial_result["closer"] = object2
                        else:
                            spatial_result["closer"] = "unknown"
                        logger.debug(f"Closer to camera from 2D bbox proxy (3D depth diff too small): {object1} cy={cy1:.1f}, {object2} cy={cy2:.1f}, result={spatial_result['closer']}")
                else:
                    # 3D centers not available, use 2D Y proxy
                    cy_diff_closer = abs(cy2 - cy1)
                    min_2d_y_separation_closer = 10.0
                    
                    if cy_diff_closer >= min_2d_y_separation_closer:
                        if cy1 > cy2:  # obj1 is lower in image (closer to camera)
                            spatial_result["closer"] = object1
                        elif cy1 < cy2:  # obj2 is lower in image (closer to camera)
                            spatial_result["closer"] = object2
                    else:
                        spatial_result["closer"] = "unknown"
                    logger.debug(f"Closer to camera from 2D bbox proxy (3D not available): {object1} cy={cy1:.1f}, {object2} cy={cy2:.1f}, result={spatial_result['closer']}")
                
                return spatial_result
            
            # Fallback: If 2D bbox not available, return unknown (3D pose not reliable for viewer perspective)
            logger.debug(f"2D bbox not available for {object1} and {object2}, returning unknown relationships")
            return {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown", "closer": "unknown"}
            
        except Exception as e:
            logger.error(f"Error calculating spatial relationship from annotations for {object1} and {object2}: {e}")
            return {"left_right": "unknown", "above_below": "unknown", "front_behind": "unknown", "closer": "unknown"}
    
    def get_spatial_answer(self, question_type: str, target_object: str, reference_object: str, 
                          spatial_relationship: Dict[str, str]) -> str:
        """Get the correct answer for spatial questions based on spatial relationship"""
        if question_type == 'spatial_left_of':
            left_right = spatial_relationship.get('left_right', 'unknown')
            if left_right == 'left':
                return 'left'  # target_object is left of reference_object
            elif left_right == 'right':
                return 'right'  # target_object is right of reference_object
            else:
                return 'unknown'
        elif question_type == 'spatial_right_of':
            left_right = spatial_relationship.get('left_right', 'unknown')
            if left_right == 'right':
                return 'right'  # target_object is right of reference_object
            elif left_right == 'left':
                return 'left'  # target_object is left of reference_object
            else:
                return 'unknown'
        elif question_type == 'spatial_above':
            above_below = spatial_relationship.get('above_below', 'unknown')
            if above_below == 'above':
                return 'above'  # target_object is above reference_object
            elif above_below == 'below':
                return 'below'  # target_object is below reference_object
            else:
                return 'unknown'
        elif question_type == 'spatial_below':
            above_below = spatial_relationship.get('above_below', 'unknown')
            if above_below == 'below':
                return 'below'  # target_object is below reference_object
            elif above_below == 'above':
                return 'above'  # target_object is above reference_object
            else:
                return 'unknown'
        elif question_type == 'spatial_in_front':
            front_behind = spatial_relationship.get('front_behind', 'unknown')
            if front_behind == 'front':
                return 'front'  # target_object is in front of reference_object
            elif front_behind == 'behind':
                return 'behind'  # target_object is behind reference_object
            else:
                return 'unknown'
        elif question_type == 'spatial_behind':
            front_behind = spatial_relationship.get('front_behind', 'unknown')
            if front_behind == 'behind':
                return 'behind'  # target_object is behind reference_object
            elif front_behind == 'front':
                return 'front'  # target_object is in front of reference_object
            else:
                return 'unknown'
        elif question_type == 'spatial_left_right':
            left_right = spatial_relationship.get('left_right', 'unknown')
            return left_right
        elif question_type == 'spatial_above_below':
            above_below = spatial_relationship.get('above_below', 'unknown')
            return above_below
        elif question_type == 'spatial_front_behind':
            front_behind = spatial_relationship.get('front_behind', 'unknown')
            return front_behind
        elif question_type == 'spatial_closer_to_camera':
            # Return the object name that is closer to camera
            closer = spatial_relationship.get('closer', 'unknown')
            if closer == 'unknown':
                return 'unknown'
            # The 'closer' value is already the object name (object1 or object2)
            return closer
        elif question_type == 'spatial_distance':
            # For spatial distance, we need to compare all objects and return the closest
            # This is handled differently - we need to check which object in available_objects
            # is closest to the reference_object, then return that object
            # For now, return unknown as this requires additional context
            return 'unknown'
        else:
            return 'unknown'
