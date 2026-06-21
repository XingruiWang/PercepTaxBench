#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import re
import sys
import math
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
from itertools import combinations
from copy import deepcopy

# Add the modules directory to the path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.qa_modules.taxonomy_utils import TaxonomyUtils
from modules.qa_modules.spatial_utils import SpatialUtils
from modules.qa_modules.visualization_utils import VisualizationUtils
from modules.qa_modules.question_templates import get_question_template
from modules.qa_modules.cot_reasoning_utils import CoTReasoningGenerator
from modules.qa_modules.data_loading_utils import DataLoadingUtils
from modules.qa_modules.unified_qa_generation_utils import UnifiedQAGenerationUtils
from modules.qa_modules.image_processing_utils import ImageProcessingUtils
from modules.qa_modules.object_utils import ObjectUtils
from modules.qa_modules.question_generation_utils import QuestionGenerationUtils
from modules.qa_modules.annotation_processing_utils import AnnotationProcessingUtils
from modules.qa_modules.question_type_grouping import get_simplified_question_type
from refresh_reasoning import refresh_reasoning as regenerate_reasoning

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GLOBAL_BASELINE_FILE = Path(__file__).resolve().parents[3] / "expected_sim_scene_count.json"

class SimImageQAGenerator:
    def __init__(
        self,
        images_dir: str,
        output_dir: str,
        taxonomy_dir: str = None,
        expected_scene_count: Optional[int] = None,
        allow_scene_drop: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.expected_scene_count = expected_scene_count
        self.allow_scene_drop = allow_scene_drop
        self.baseline_path = GLOBAL_BASELINE_FILE
        
        # Initialize taxonomy utils
        if taxonomy_dir is None:
            taxonomy_dir = Path(__file__).resolve().parents[2] / "taxonomy"
        self.taxonomy_utils = TaxonomyUtils(Path(taxonomy_dir))
        self.spatial_utils = SpatialUtils()
        self.visualization_utils = VisualizationUtils()
        self.image_processing_utils = ImageProcessingUtils()
        self.object_utils = ObjectUtils(taxonomy_utils=self.taxonomy_utils)
        self.question_generation_utils = QuestionGenerationUtils(taxonomy_utils=self.taxonomy_utils, object_utils=self.object_utils)
        self.data_loading_utils = DataLoadingUtils()
        self.annotation_processing_utils = AnnotationProcessingUtils()
        self.cot_generator = CoTReasoningGenerator(taxonomy_utils=self.taxonomy_utils)
        self.unified_qa_generation_utils = UnifiedQAGenerationUtils(
            taxonomy_utils=self.taxonomy_utils,
            object_utils=self.object_utils,
            question_generation_utils=self.question_generation_utils,
            data_loading_utils=self.data_loading_utils
        )
        self.image_quality_ratings = self.data_loading_utils.load_image_quality_ratings()
        self.min_lighting_exposure = 5.0
        self._missing_quality_keys: Set[str] = set()

        # Cache QA space data and enrich it with physical property coverage for sim objects
        self.qa_space_data = self.data_loading_utils.load_qa_space_data() or {}
        self._ensure_physical_property_mapping()
        
        # Load sim scene data
        self.sm_to_taxonomy = self.data_loading_utils.load_sm_to_human_mapping()  # SM_name -> taxonomy_name
        self.scene_object_categories = self.data_loading_utils.load_scene_object_categories()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized Sim Image QA Generator with modular components")

    @staticmethod
    def _normalize_object_name(name: Optional[str]) -> str:
        return (name or "").strip().lower()

    @staticmethod
    def _augment_question_text(text: Optional[str]) -> Optional[str]:
        if not text or not isinstance(text, str):
            return text
        stripped = text.strip()
        if not stripped:
            return stripped
        if stripped.lower().startswith("in this scene"):
            return stripped
        first_char = stripped[0]
        remainder = stripped[1:] if len(stripped) > 1 else ""
        lowered = (first_char.lower() + remainder) if first_char.isupper() else stripped
        return f"In this scene, {lowered}"

    def _ensure_physical_property_mapping(self) -> None:
        """
        Ensure the QA space data exposes a `physical_property` question type by
        synthesizing it from taxonomy clusters when the QA analysis did not emit it.
        """
        mappings = self.qa_space_data.setdefault("question_answer_mappings", {})
        if "physical_property" in mappings:
            return

        taxonomy_clusters = getattr(self.taxonomy_utils, "taxonomy_clusters", None)
        if not taxonomy_clusters:
            logger.warning("Taxonomy clusters unavailable; cannot synthesize physical_property QA mapping.")
            return

        physical_taxonomy = taxonomy_clusters.get("final_taxonomy_physical_properties", {}) or {}
        if not isinstance(physical_taxonomy, dict):
            logger.warning("Unexpected physical taxonomy structure; physical_property questions remain disabled.")
            return

        if "physical_properties" in physical_taxonomy and isinstance(physical_taxonomy["physical_properties"], dict):
            cluster_source = physical_taxonomy["physical_properties"].items()
        else:
            cluster_source = physical_taxonomy.items()

        physical_objects: Set[str] = set()
        for cluster_name, cluster_data in cluster_source:
            if not cluster_data or not isinstance(cluster_data, dict):
                continue
            if cluster_name and "no-physical" in cluster_name.lower():
                continue
            for obj in (cluster_data.get("objects") or []):
                norm = self._normalize_object_name(obj)
                if norm:
                    physical_objects.add(norm)

        if not physical_objects:
            logger.warning("No physical property objects found in taxonomy clusters; physical_property questions remain disabled.")
            return

        mappings["physical_property"] = {
            "description": "Objects with non-void physical property clusters",
            "filter_criteria": ["final_taxonomy_physical_properties != void"],
            "sm_valid_objects": sorted(physical_objects),
            "openimages_valid_objects": [],
        }


    def _extract_objects_from_scene(self, scene_path: Path) -> Tuple[
        List[str], Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, Any]
    ]:
        """Extract visible objects and their pose data from a sim scene
        
        Returns:
            Tuple of (objects, object_poses, taxonomy_to_sm_names mapping, scene_metadata)
        """
        # Extract objects and poses using annotation processing utils
        # Pass sm_to_taxonomy directly (simplified - no base name matching)
        objects, object_poses, taxonomy_to_sm_names, scene_metadata = self.annotation_processing_utils.extract_objects_from_sim_scene(
            scene_path, self.sm_to_taxonomy
        )
        
        return objects, object_poses, taxonomy_to_sm_names, scene_metadata
    
    def _get_scene_category_from_path(self, scene_path: Path) -> str:
        """Extract scene category from path (e.g., 'Diner50 2')"""
        # Get the parent directory name as the scene category
        scene_category = scene_path.parent.name
        return scene_category
    
    def _filter_objects_by_scene(self, objects: List[str], scene_name: str) -> List[str]:
        """Filter objects to only include those that belong to the scene"""
        if scene_name not in self.scene_object_categories:
            return objects
        
        scene_objects = set()
        scene_categories = self.scene_object_categories[scene_name]
        
        # Get all objects that belong to this scene
        for category, obj_list in scene_categories.items():
            for obj in obj_list:
                # Convert SM name to taxonomy name
                taxonomy_name = self.sm_to_taxonomy.get(obj, obj)
                scene_objects.add(taxonomy_name)
        
        # Filter objects to only include those in the scene
        filtered_objects = [obj for obj in objects if obj in scene_objects]
        
        logger.info(f"Filtered {len(objects)} objects to {len(filtered_objects)} scene-relevant objects for {scene_name}")
        return filtered_objects
    
    def _get_base_scene_name(self, scene_category: str) -> str:
        """Extract base scene name for filtering (e.g., 'Diner50 2' -> 'Diner50')"""
        return ''.join(c for c in scene_category if not c.isdigit()).strip()
    
    def _generate_questions_for_objects(self, objects: List[str], scene_id: str, object_poses: Dict[str, Dict[str, Any]], scene_path: Path = None) -> List[Dict[str, Any]]:
        """Generate questions for a set of objects with pose data"""
        # TEMPORARILY DISABLED: Spatial question generation (3D pose data not ready)
        # Only generate taxonomy-based questions for now
        questions = []
        
        # Initialize CoT reasoning generator
        cot_generator = CoTReasoningGenerator(taxonomy_utils=self.taxonomy_utils)
        
        # TEMPORARILY COMMENTED OUT: Spatial question generation
        # Spatial questions require 3D pose data which is not ready yet
        # Only generate spatial questions for sim images as they don't require QA space matching
        # spatial_question_types = [
        #     'spatial_left_of',
        #     'spatial_right_of', 
        #     'spatial_above',
        #     'spatial_below',
        #     'spatial_front_behind'
        # ]
        # 
        # for question_type in spatial_question_types:
        #     try:
        #         if len(objects) < 2:
        #             continue
        #         
        #         template_questions = self._generate_spatial_questions(
        #             question_type, objects, scene_id, cot_generator, object_poses, scene_path
        #         )
        #         questions.extend(template_questions)
        #         
        #     except Exception as e:
        #         logger.warning(f"Error generating spatial question {question_type}: {e}")
        #         continue
        
        logger.debug(f"Skipping spatial question generation - using unified QA generation instead")
        return questions
    
    def _generate_questions_for_type(self, question_type: str, objects: List[str], 
                                    scene_id: str, cot_generator: CoTReasoningGenerator, 
                                    object_poses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate questions for a specific question type with pose data"""
        # TEMPORARILY DISABLED: Not used for sim images (using unified QA generation instead)
        logger.debug(f"_generate_questions_for_type called but not used - using unified QA generation")
        return []
    
    # TEMPORARILY DISABLED: Spatial question generation (3D pose data not ready)
    # def _generate_spatial_questions(self, question_type: str, objects: List[str], 
    #                                scene_id: str, cot_generator: CoTReasoningGenerator,
    #                                object_poses: Dict[str, Dict[str, Any]],
    #                                scene_path: Path = None) -> List[Dict[str, Any]]:
    #     """Generate spatial questions using pose data"""
    #     questions = []
    #     ...
    #     return questions
    
    
    
    def _apply_filters(self, objects: List[str], filter_conditions: Dict[str, Any]) -> List[str]:
        """Apply filter conditions to objects"""
        if not filter_conditions:
            return objects
        
        filtered_objects = []
        
        for obj in objects:
            # Check if object meets filter conditions
            if self._object_meets_conditions(obj, filter_conditions):
                filtered_objects.append(obj)
        
        return filtered_objects
    
    def _object_meets_conditions(self, obj: str, conditions: Dict[str, Any]) -> bool:
        """Check if an object meets the given conditions"""
        # Simplified condition checking - in a real implementation,
        # this would check taxonomy clusters, properties, etc.
        return True  # For now, accept all objects

    def _lookup_image_quality(self, scene_path: Path) -> Tuple[Optional[float], Optional[str]]:
        """Return lighting exposure score and the key used for lookup, if available"""
        if not self.image_quality_ratings:
            return None, None

        scene_name = scene_path.parent.name
        room_id = scene_path.name

        candidate_keys = [
            f"{scene_name}/{room_id}",
            f"{scene_name}_{room_id}",
        ]

        for key in candidate_keys:
            metrics = self.image_quality_ratings.get(key)
            if metrics:
                lighting_value = metrics.get("LightingExposure")
                if isinstance(lighting_value, (int, float)):
                    return float(lighting_value), key

        return None, None
    
    def process_scene(self, scene_path: Path) -> Dict[str, Any]:
        """Process a single sim scene and generate QA data"""
        scene_id = scene_path.name
        scene_category = self._get_scene_category_from_path(scene_path)
        
        logger.info(f"Processing scene: {scene_id} (scene_category: {scene_category})")
        
        # Extract objects and their pose data from the scene
        objects, object_poses, taxonomy_to_sm_names, scene_metadata = self._extract_objects_from_scene(scene_path)
        
        if not objects:
            logger.warning(f"No objects found for scene {scene_id}")
            return None
        
        # Filter objects to only include those relevant to the scene
        base_scene_name = self._get_base_scene_name(scene_category)
        objects = self._filter_objects_by_scene(objects, base_scene_name)
        
        # Filter out structural/background objects by name (imported from filter_utils)
        from modules.qa_modules.filter_utils import STRUCTURAL_OBJECTS_TO_FILTER
        objects = [obj for obj in objects if obj.lower() not in STRUCTURAL_OBJECTS_TO_FILTER]
        logger.info(f"After name-based filtering: {len(objects)} objects")
        
        # Filter out objects in void clusters (same as real images)
        from modules.qa_modules.filter_utils import should_exclude_object_from_qa
        objects = [obj for obj in objects if not should_exclude_object_from_qa(obj, self.taxonomy_utils)]
        logger.info(f"After void cluster filtering: {len(objects)} objects")
        
        # Unified filtering: segmentation visibility, size, and depth in one pass
        objects, validated_bboxes = self._filter_objects_by_visibility_size_and_depth(scene_path, objects, object_poses, taxonomy_to_sm_names)
        
        if len(objects) < 2:
            logger.warning(f"Not enough objects ({len(objects)}) for QA generation in scene {scene_id}")
            return None
        
        # Don't generate questions here - will be generated after grouping to check for conflicts per group
        return {
            'scene_id': scene_id,
            'scene_category': scene_category,
            'objects': objects,
            'object_poses': object_poses,
            'taxonomy_to_sm_names': taxonomy_to_sm_names,
            'validated_bboxes': validated_bboxes,  # Store validated bboxes for reuse
            'scene_metadata': scene_metadata,
            'questions': []  # Will be populated after grouping
        }
    
    def generate_qa_benchmark(self, max_images: int = None, refresh_reasoning_enabled: bool = True) -> Dict[str, Any]:
        """Generate QA benchmark for all sim scene images (views)"""
        logger.info("Starting sim image QA benchmark generation")
        
        # Find all scene directories and zip files
        scene_dirs = []
        
        for item in self.images_dir.iterdir():
            if item.is_dir():
                # Each scene directory contains multiple view directories
                for view_dir in item.iterdir():
                    if view_dir.is_dir():
                        # PRIMARY: Check for new processed format (scene_annotations_split.json)
                        # FALLBACK: Check for legacy format (lit.png)
                        if (view_dir / "scene_annotations_split.json").exists():
                            scene_dirs.append(view_dir)
                        elif (view_dir / "lit.png").exists():
                            scene_dirs.append(view_dir)
        
        logger.info(f"Found {len(scene_dirs)} sim scene views (images)")
        
        if max_images:
            # Randomly sample max_images from all scenes for diversity
            random.shuffle(scene_dirs)
            scene_dirs = scene_dirs[:max_images]
            logger.info(f"Randomly sampling {max_images} images (views) from all scenes")
        
        # Process each scene
        all_questions = []
        scene_statistics = []
        processed_scenes = 0
        
        try:
            for image_path in scene_dirs:
                try:
                    # Skip images without required annotation files
                    # Check for new processed format or legacy format
                    has_processed_format = (image_path / "scene_annotations_split.json").exists()
                    has_legacy_format = (image_path / "seenable_obj_dict.json").exists()
                    
                    if not has_processed_format and not has_legacy_format:
                        logger.info(f"Skipping image {image_path.name} - no annotation files found")
                        continue
                    
                    # Include parent directory name to ensure uniqueness across scenes
                    scene_id = f"{image_path.parent.name}_{image_path.name}"  # Make unique across scenes
                    scene_path = image_path  # Store as scene_path for later use

                    lighting_exposure, rating_key = self._lookup_image_quality(image_path)
                    if lighting_exposure is not None and lighting_exposure < self.min_lighting_exposure:
                        logger.info(
                            f"Skipping image {scene_id} due to LightingExposure {lighting_exposure} "
                            f"(< {self.min_lighting_exposure}) from rating key {rating_key}"
                        )
                        continue
                    if lighting_exposure is None:
                        base_key = f"{image_path.parent.name}/{image_path.name}"
                        if base_key not in self._missing_quality_keys:
                            self._missing_quality_keys.add(base_key)
                            logger.debug(f"No lighting exposure rating found for sim view {base_key}")

                    scene_data = self.process_scene(image_path)
                    
                    # Skip if scene_data is None (no objects found)
                    if not scene_data:
                        logger.info(f"Skipping image {image_path.name} - no objects found after filtering")
                        continue
                    
                    questions = scene_data.get('questions', [])
                    # Don't extend all_questions here - wait to see if grouping happens
                    
                    # Copy images with bounding boxes using ImageProcessingUtils
                    objects = scene_data.get('objects', [])
                    object_poses = scene_data.get('object_poses', {})
                    taxonomy_to_sm_names = scene_data.get('taxonomy_to_sm_names', {})
                    validated_bboxes = scene_data.get('validated_bboxes', {})
                    scene_metadata = scene_data.get('scene_metadata', {})
                    
                    # Double-check: Skip if no objects after filtering
                    if not objects or len(objects) == 0:
                        logger.info(f"Skipping image {image_path.name} - no objects after filtering")
                        continue
                    
                    # Objects are already filtered by visibility and size
                    # Just ensure we have enough objects with valid bboxes
                    objects_with_bbox = [obj for obj in objects if validated_bboxes.get(obj)]
                    if len(objects_with_bbox) < 3:
                        logger.warning(f"Not enough objects with valid bboxes ({len(objects_with_bbox)}) for scene {scene_id}, skipping")
                        continue
                    
                    logger.info(f"Objects after filtering: {len(objects_with_bbox)}")
                    
                    # Get unique object categories (human-readable names)
                    unique_objects = len(objects_with_bbox)
                    
                    # Create images directory in output
                    images_dir = self.output_dir / "images"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Group objects: keep all together if ≤6, split into groups of 6 if >6
                    if unique_objects <= 6:
                        logger.info(f"Scene {scene_id} has {unique_objects} objects (≤6) - keeping all together")
                        object_groups = [objects_with_bbox]
                    else:
                        # Split objects into groups (max 6 objects per group)
                        object_groups = self._split_objects_into_groups(objects_with_bbox, max_objects_per_group=6)
                        logger.info(f"Split {len(objects_with_bbox)} objects into {len(object_groups)} groups for scene {scene_id}")
                    
                    
                    # Generate bbox images for each group and collect objects with bboxes
                    # First pass: create bbox images and collect all objects with valid bboxes
                    all_group_questions = []
                    group_bbox_results = {}  # group_idx -> (objects_with_bboxes, group_output_dir)
                    orphaned_objects = []  # Objects from groups with < 3 bboxes that need redistribution
                    
                    for group_idx, group_objects in enumerate(object_groups):
                        # Process all groups, even if they have fewer than 6 objects (last group)
                        if len(group_objects) == 0:
                            continue
                        
                        group_image_path = f"{scene_id}_group{group_idx}"
                        group_output_dir = images_dir / group_image_path
                        
                        # Create bbox image with only the objects in this group
                        objects_with_bboxes_in_group = self.image_processing_utils.copy_images_with_bbox_group(
                            scene_id, image_path, images_dir, group_objects, group_idx,
                            object_poses, taxonomy_to_sm_names, validated_bboxes, scene_metadata
                        )
                        
                        # Handle None return (when image file not found)
                        if objects_with_bboxes_in_group is None:
                            logger.warning(f"No image file found for {scene_id}_group{group_idx}, skipping this group")
                            if group_output_dir.exists():
                                import shutil
                                shutil.rmtree(group_output_dir)
                                logger.info(f"Deleted empty group directory for {scene_id}_group{group_idx}")
                            continue
                        
                        logger.info(f"DEBUG: objects_with_bboxes_in_group = {objects_with_bboxes_in_group}")
                        
                        if len(objects_with_bboxes_in_group) < 1:
                            logger.warning(f"No bboxes created for {scene_id}_group{group_idx}, skipping this group")
                            # Delete the directory since no objects were detected
                            if group_output_dir.exists():
                                import shutil
                                shutil.rmtree(group_output_dir)
                                logger.info(f"Deleted empty group directory for {scene_id}_group{group_idx}")
                            continue
                        
                        # Store results for groups with ≥3 objects
                        if len(objects_with_bboxes_in_group) >= 3:
                            group_bbox_results[group_idx] = (objects_with_bboxes_in_group, group_output_dir)
                        else:
                            # Collect objects from groups with < 3 bboxes for redistribution
                            logger.debug(f"Group {scene_id}_group{group_idx} has {len(objects_with_bboxes_in_group)} objects with bboxes - will try to redistribute")
                            orphaned_objects.extend(objects_with_bboxes_in_group)
                            # Delete the directory since we don't have enough objects (will recreate if redistribution succeeds)
                            if group_output_dir.exists():
                                import shutil
                                shutil.rmtree(group_output_dir)
                                logger.info(f"Deleted group directory for {scene_id}_group{group_idx} (insufficient objects, will try redistribution)")
                    
                    # Redistribution phase: try to add orphaned objects to existing valid groups
                    if orphaned_objects:
                        logger.info(f"Attempting to redistribute {len(orphaned_objects)} objects from groups with < 3 bboxes")
                        
                        # Track which objects were successfully redistributed
                        redistributed_objects = set()
                        
                        # Find valid groups (≥3 objects) that have space (< 6 objects)
                        valid_groups_with_space = []
                        for group_idx, (objects_with_bboxes, _) in group_bbox_results.items():
                            if len(objects_with_bboxes) >= 3 and len(objects_with_bboxes) < 6:
                                valid_groups_with_space.append((group_idx, objects_with_bboxes))
                        
                        # Sort by current size (fill smaller groups first for better distribution)
                        valid_groups_with_space.sort(key=lambda x: len(x[1]))
                        
                        # Redistribute orphaned objects to valid groups with space
                        for orphan_obj in orphaned_objects:
                            redistributed = False
                            for group_idx, objects_with_bboxes in valid_groups_with_space:
                                if len(objects_with_bboxes) < 6:
                                    # Recreate bbox image with added object
                                    group_objects = object_groups[group_idx].copy()
                                    if orphan_obj not in group_objects:
                                        group_objects.append(orphan_obj)
                                    
                                    # Delete old bbox image and recreate
                                    group_output_dir = group_bbox_results[group_idx][1]
                                    if group_output_dir.exists():
                                        import shutil
                                        shutil.rmtree(group_output_dir)
                                    
                                    # Recreate with updated object list
                                    updated_bboxes = self.image_processing_utils.copy_images_with_bbox_group(
                                        scene_id, image_path, images_dir, group_objects, group_idx,
                                        object_poses, taxonomy_to_sm_names, validated_bboxes, scene_metadata
                                    )
                                    
                                    # Only accept if the object actually got a bbox
                                    if updated_bboxes and orphan_obj in updated_bboxes:
                                        group_bbox_results[group_idx] = (updated_bboxes, group_output_dir)
                                        # Update the list in valid_groups_with_space for next iteration
                                        objects_with_bboxes.clear()
                                        objects_with_bboxes.extend(updated_bboxes)
                                        redistributed_objects.add(orphan_obj)
                                        redistributed = True
                                        logger.debug(f"Redistributed {orphan_obj} to group {group_idx} (now {len(updated_bboxes)} objects)")
                                        break
                                    else:
                                        # Restore original bbox image if redistribution failed
                                        if group_output_dir.exists():
                                            import shutil
                                            shutil.rmtree(group_output_dir)
                                        # Recreate original
                                        original_bboxes = self.image_processing_utils.copy_images_with_bbox_group(
                                            scene_id, image_path, images_dir, object_groups[group_idx], group_idx,
                                            object_poses, taxonomy_to_sm_names, validated_bboxes, scene_metadata
                                        )
                                        if original_bboxes:
                                            group_bbox_results[group_idx] = (original_bboxes, group_output_dir)
                            
                            if not redistributed:
                                logger.debug(f"Could not redistribute {orphan_obj} - no groups with space or bbox extraction failed")
                        
                        if redistributed_objects:
                            logger.info(f"Successfully redistributed {len(redistributed_objects)} objects to existing groups")
                    
                    # Process valid groups (≥3 objects with bboxes) for question generation
                    # All groups in group_bbox_results already have ≥3 objects (filtered above)
                    valid_group_results = {}
                    for group_idx, (objects_with_bboxes_in_group, group_output_dir) in group_bbox_results.items():
                        # Groups already filtered to have ≥3 objects, so process all
                        valid_group_results[group_idx] = (objects_with_bboxes_in_group, group_output_dir)
                    
                    # Skip scene if no valid groups (≥3 objects) remaining
                    if not valid_group_results:
                        logger.warning(f"No valid groups (≥3 objects with bboxes) remaining for scene {scene_id}, skipping scene")
                        continue
                    
                    # Process only valid groups (≥3 objects)
                    for group_idx, (objects_with_bboxes_in_group, group_output_dir) in valid_group_results.items():
                        scene_rel_path = scene_path.relative_to(self.images_dir)
                        
                        # Generate questions FOR THIS GROUP ONLY
                        # Use ONLY objects that have bboxes for question generation
                        # Get minimal pose data (only 2D bbox) for objects in this group
                        group_object_poses = {obj: object_poses.get(obj, {}) for obj in objects_with_bboxes_in_group if obj in object_poses}
                        
                        # Generate questions for this group's objects using the same pipeline as real images
                        # Use cached QA space data enriched with physical property mapping
                        group_questions = self.unified_qa_generation_utils.generate_qa_from_space(
                            f"{scene_id}_group{group_idx}", 
                            objects_with_bboxes_in_group, 
                            self.qa_space_data,
                            use_sm_valid_only=True,  # For sim images, only use sm_valid_objects
                            max_questions_per_image=5  # Sim images: cap at 8 questions per group
                        )
                        
                        # Post-process: update each question to include ONLY objects that have bboxes
                        def _to_float(value: Any) -> Any:
                            try:
                                if isinstance(value, bool):
                                    return value
                                return float(value)
                            except Exception:
                                try:
                                    return float(value.item())  # numpy scalar
                                except Exception:
                                    return value

                        def _to_float_list(values: Any) -> Optional[List[Any]]:
                            if not isinstance(values, (list, tuple)):
                                return None
                            result: List[Any] = []
                            for item in values:
                                result.append(_to_float(item))
                            return result

                        def _convert_nested_mapping(mapping: Any) -> Any:
                            if isinstance(mapping, dict):
                                return {key: _convert_nested_mapping(val) for key, val in mapping.items()}
                            if isinstance(mapping, (list, tuple)):
                                return [_convert_nested_mapping(val) for val in mapping]
                            return _to_float(mapping)

                        def _build_bbox_metadata(obj_name: str) -> Optional[Dict[str, Any]]:
                            bbox_info = validated_bboxes.get(obj_name) or {}
                            bbox = bbox_info.get("bbox")
                            if not bbox or len(bbox) < 4:
                                return None
                            bbox_values = _to_float_list(bbox)
                            if not bbox_values or len(bbox_values) < 4:
                                return None
                            x1, y1, x2, y2 = bbox_values[:4]
                            center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                            metadata = {
                                "bbox": bbox_values,
                                "center": center,
                                "segment_id": _to_float(bbox_info.get("segment_id")) if bbox_info.get("segment_id") is not None else None,
                                "bbox_area": _to_float(bbox_info.get("bbox_area")),
                            }
                            return metadata

                        def _build_pose_metadata(obj_name: str) -> Optional[Dict[str, Any]]:
                            pose = object_poses.get(obj_name) or {}
                            if not pose:
                                return None
                            yaw = None
                            rotation = pose.get("rotation")
                            if isinstance(rotation, (list, tuple)) and len(rotation) >= 2:
                                try:
                                    yaw = float(rotation[1])
                                except Exception:
                                    yaw = None
                            rotation_vals = _to_float_list(rotation) if isinstance(rotation, (list, tuple)) else None
                            bbox_2d = pose.get("bbox_2d")
                            bbox_2d_vals = _to_float_list(bbox_2d) if isinstance(bbox_2d, (list, tuple)) else None
                            if bbox_2d_vals and len(bbox_2d_vals) >= 4:
                                x1, y1, x2, y2 = bbox_2d_vals[:4]
                                pose_center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                            else:
                                pose_center = None
                            return {
                                "location": _to_float_list(pose.get("location")) if isinstance(pose.get("location"), (list, tuple)) else _convert_nested_mapping(pose.get("location")),
                                "rotation": rotation_vals,
                                "yaw_deg": yaw,
                                "bbox_2d": bbox_2d_vals,
                                "bbox_center": pose_center,
                                "bounds": _convert_nested_mapping(pose.get("bounds")),
                                "obb": _convert_nested_mapping(pose.get("obb")),
                                "aabb": _convert_nested_mapping(pose.get("aabb")),
                            }

                        for q in group_questions:
                            # Update objects and choices to include ONLY objects that have bboxes
                            # Use list() to create a completely new list instance
                            q['objects'] = list(objects_with_bboxes_in_group)
                            q['choices'] = list(objects_with_bboxes_in_group)
                            # Ensure image_path points to actual images directory
                            q['image_path'] = f"images/{scene_id}_group{group_idx}/bbox.jpg"
                            # Ensure image_id is set (some questions might have scene_id instead)
                            if 'image_id' not in q or not q.get('image_id'):
                                q['image_id'] = f"{scene_id}_group{group_idx}"
                            q['source_scene_id'] = scene_id
                            q['source_scene_path'] = str(scene_rel_path)
                            q['group_index'] = group_idx

                            bbox_metadata = {}
                            pose_metadata = {}
                            for obj_name in objects_with_bboxes_in_group:
                                bbox_info = _build_bbox_metadata(obj_name)
                                if bbox_info:
                                    bbox_metadata[obj_name] = bbox_info
                                pose_info = _build_pose_metadata(obj_name)
                                if pose_info:
                                    pose_metadata[obj_name] = pose_info
                            if bbox_metadata:
                                q['bbox_metadata'] = bbox_metadata
                            if pose_metadata:
                                q['pose_metadata'] = pose_metadata
                            
                            # Update question text to have proper multiple choice format
                            if 'Options:' in q['question']:
                                choices_str = ', '.join([str(i+1) for i in range(len(objects_with_bboxes_in_group))])
                                q['question'] = q['question'].split('Options:')[0] + f"Options: {choices_str}"
                        
                        # Optionally generate spatial questions using 3D pose data if available
                        spatial_question_types = [
                            'spatial_left_right',
                            'spatial_above_below',
                            'spatial_front_behind',
                            'spatial_closer_to_camera'
                        ]
                        spatial_questions_candidates = []
                        existing_question_texts = {q.get('question', '').strip() for q in group_questions if q.get('question')}
                        
                        for obj1, obj2 in combinations(objects_with_bboxes_in_group, 2):
                            pose1 = group_object_poses.get(obj1, {})
                            pose2 = group_object_poses.get(obj2, {})
                            if not pose1.get('location') or not pose2.get('location'):
                                # Require 3D locations for both objects
                                continue
                            
                            for spatial_type in spatial_question_types:
                                try:
                                    candidate_list = self.question_generation_utils.generate_spatial_questions_sim(
                                        spatial_type,
                                        [obj1, obj2],
                                        f"{scene_id}_group{group_idx}",
                                        self.cot_generator,
                                        group_object_poses,
                                        image_path
                                    )
                                except Exception as e:
                                    logger.debug(f"Failed to generate spatial question {spatial_type} for {obj1}, {obj2}: {e}")
                                    continue
                                
                                for candidate in candidate_list:
                                    question_text = candidate.get('question', '').strip()
                                    if not question_text or question_text in existing_question_texts:
                                        continue
                                    
                                    # Align metadata with existing questions (choices include full group for color mapping)
                                    candidate['objects'] = list(objects_with_bboxes_in_group)
                                    candidate['choices'] = list(objects_with_bboxes_in_group)
                                    candidate['image_path'] = f"images/{scene_id}_group{group_idx}/bbox.jpg"
                                    candidate['image_id'] = f"{scene_id}_group{group_idx}"
                                    candidate['spatial_pair'] = [obj1, obj2]
                                    candidate['source_scene_id'] = scene_id
                                    candidate['source_scene_path'] = str(scene_rel_path)
                                    candidate['group_index'] = group_idx

                                    bbox_metadata = {}
                                    pose_metadata = {}
                                    for obj_name in objects_with_bboxes_in_group:
                                        bbox_info = _build_bbox_metadata(obj_name)
                                        if bbox_info:
                                            bbox_metadata[obj_name] = bbox_info
                                        pose_info = _build_pose_metadata(obj_name)
                                        if pose_info:
                                            pose_metadata[obj_name] = pose_info
                                    if bbox_metadata:
                                        candidate['bbox_metadata'] = bbox_metadata
                                    if pose_metadata:
                                        candidate['pose_metadata'] = pose_metadata
                                    
                                    spatial_questions_candidates.append(candidate)
                                    existing_question_texts.add(question_text)
                        
                        if spatial_questions_candidates and random.random() < 0.6:
                            random.shuffle(spatial_questions_candidates)
                            max_spatial_per_group = min(2, len(spatial_questions_candidates))
                            selected_spatial_questions = spatial_questions_candidates[:max_spatial_per_group]
                            logger.info(f"Adding {len(selected_spatial_questions)} spatial question(s) for {scene_id}_group{group_idx}")
                            group_questions.extend(selected_spatial_questions)
                        else:
                            if spatial_questions_candidates:
                                logger.debug(f"Spatial questions generated but not selected for {scene_id}_group{group_idx} (random skip or none)")
                        
                        if group_questions:
                            random.shuffle(group_questions)
                            drop_count = math.floor(len(group_questions) * 0.2)
                            max_drops = max(0, len(group_questions) - 1)
                            drop_count = min(drop_count, max_drops)
                            if drop_count > 0:
                                dropped = group_questions[:drop_count]
                                group_questions = group_questions[drop_count:]
                                logger.info(
                                    f"Randomly dropped {len(dropped)} question(s); retaining {len(group_questions)} for {scene_id}_group{group_idx}"
                                )
                        logger.info(f"Generated {len(group_questions)} questions for {scene_id}_group{group_idx} (bbox objects: {len(objects_with_bboxes_in_group)}, original group: {len(group_objects)})")
                        all_group_questions.extend(group_questions)
                    
                    # Use the grouped questions
                    questions = all_group_questions
                    
                    # Only add scenes that have questions generated
                    if len(questions) > 0:
                        all_questions.extend(questions)
                        
                        # Determine the final scene_id (could be grouped or not)
                        # Use the same unique naming scheme as scene_id above
                        final_scene_id = f"{image_path.parent.name}_{image_path.name}"
                        if unique_objects > 6:
                            # Count actual groups created
                            actual_groups = len([q for q in questions])  # Approximate
                            final_scene_id = f"{image_path.parent.name}_{image_path.name}"  # Keep original for stats
                        
                        scene_statistics.append({
                            'scene_id': final_scene_id,
                            'scene_category': scene_data.get('scene_category', 'unknown'),
                            'num_objects': unique_objects,  # Use filtered count
                            'num_questions': len(questions),
                            'question_types': list(set(q.get('question_type', '') for q in questions))
                        })
                        
                        processed_scenes += 1
                    else:
                        logger.debug(f"Skipping scene {scene_id} - no questions generated")

                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    continue
        
        finally:
            # Clean up temporary extraction directory if it was created
            # TEMPORARILY COMMENTED OUT: Cleanup not needed when zip files are skipped
            # if temp_extract_dir is not None and temp_extract_dir.exists():
            #     import shutil
            #     logger.info(f"Cleaning up temporary extraction directory: {temp_extract_dir}")
            #     try:
            #         shutil.rmtree(temp_extract_dir)
            #         logger.info(f"Successfully removed temporary directory")
            #     except Exception as e:
            #         logger.warning(f"Failed to remove temporary directory {temp_extract_dir}: {e}")
            pass  # No cleanup needed when zip files are skipped
        
        logger.info(f"QA generation complete: {len(all_questions)} questions generated for {processed_scenes} scenes")
        
        # Save results
        self._enforce_scene_count_guard(processed_scenes)
        self._save_results(
            all_questions,
            scene_statistics,
            processed_scenes,
            refresh_reasoning_enabled,
        )
        
        return {
            'total_questions': len(all_questions),
            'total_scenes': processed_scenes,
            'questions': all_questions,
            'scene_statistics': scene_statistics
        }
    
    def _get_visible_objects_with_mapping(self, scene_path: Path, objects: List[str], object_poses: Dict[str, Dict]) -> Tuple[List[str], Dict[str, str]]:
        """
        Get objects that are visible in segmentation and create mapping between:
        taxonomy name -> SM object name, and store their colors
        
        Returns:
            Tuple of (visible_object_names, mapping of object_name -> (sm_name, color, color_int))
        """
        import json
        from modules.qa_modules.visualization_utils import VisualizationUtils
        
        seg_image_path = scene_path / "seg.png"
        if not seg_image_path.exists():
            return objects, {}  # Return all if no segmentation
        
        # Get all bboxes from segmentation
        all_bboxes = VisualizationUtils.extract_bboxes_from_segmentation(str(seg_image_path))
        seg_ids = {bbox['segment_id'] for bbox in all_bboxes}
        
        # Load object annotations to get color mapping
        object_annots_file = scene_path / "object_annots.json"
        if not object_annots_file.exists():
            return objects, {}
        
        try:
            with open(object_annots_file, 'r') as f:
                obj_data = json.load(f)
        except:
            return objects, {}
            
        # Load all_objects mapping (taxonomy -> SM names)
        all_objects_file = Path(__file__).parent / "modules" / "sim_scene_object" / "data" / "object_list_v4.json"
        taxonomy_to_sm = {}  # taxonomy name -> list of SM names in scene
        sm_to_taxonomy = {}  # reverse: SM name -> taxonomy name
        try:
            if all_objects_file.exists():
                with open(all_objects_file, 'r') as f:
                    all_objects_data = json.load(f)
                # Build reverse mapping: SM name -> taxonomy name
                for taxonomy_name, entries in all_objects_data.items():
                    for entry in entries:
                        sm_name = entry.get('object name', '')
                        if sm_name:
                            sm_to_taxonomy[sm_name] = taxonomy_name
        except:
            pass
        
        # Build mapping: for each SM object in scene, get its taxonomy name and color
        visible_objects = []
        object_mapping = {}  # taxonomy_name -> (sm_name, color, color_int) for visible objects
        
        for obj in obj_data.get('outputs', []):
            sm_name = obj.get('object_id', '')
            color = obj.get('color', None)
            if sm_name and color:
                # Get taxonomy name from mapping
                taxonomy_name = sm_to_taxonomy.get(sm_name)
                if taxonomy_name:
                    # Calculate color_int
                    if isinstance(color, list) and len(color) >= 3:
                        color_int = color[0] * 256 * 256 + color[1] * 256 + color[2]
                        # Check if this object is visible in segmentation
                        if color_int in seg_ids:
                            # This is a visible object
                            if taxonomy_name not in visible_objects:
                                visible_objects.append(taxonomy_name)
                            # Store mapping: taxonomy name -> (sm_name, color, color_int)
                            object_mapping[taxonomy_name] = (sm_name, color, color_int)
        
        return visible_objects, object_mapping
    
    def _enforce_scene_count_guard(self, processed_scenes: int) -> None:
        """Ensure scene count has not regressed below the expected minimum."""
        if self.expected_scene_count is None:
            logger.info(f"No baseline scene count recorded; accepting {processed_scenes} scenes.")
            return
        
        if self.allow_scene_drop:
            logger.info(
                f"Scene count guard disabled (--allow_scene_drop). "
                f"Generated {processed_scenes} scenes (baseline: {self.expected_scene_count})."
            )
            return
        
        if processed_scenes < self.expected_scene_count:
            raise RuntimeError(
                f"Generated {processed_scenes} scenes, which is below the expected baseline "
                f"of {self.expected_scene_count}. Aborting to prevent silent scene loss. "
                f"Use --allow_scene_drop to override if this reduction is intentional."
            )
        
        if processed_scenes > self.expected_scene_count:
            logger.info(
                f"Scene count increased from baseline {self.expected_scene_count} to {processed_scenes}. "
                f"Baseline will be updated."
            )
        else:
            logger.info(f"Scene count matches baseline ({processed_scenes}).")
    
    def _filter_objects_by_visibility_size_and_depth(self, scene_path: Path, objects: List[str], object_poses: Dict[str, Dict], taxonomy_to_sm_names: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Unified filtering: segmentation visibility and size.
        
        TEMPORARILY SIMPLIFIED: Skip depth filtering (not needed without spatial questions)
        Filters objects based on:
        1. Visible in segmentation (for legacy format) or foreground (for processed format)
        2. Bbox size (≥ 0.5% of image area, min 40x40px)
        
        Args:
            scene_path: Path to scene directory
            objects: List of taxonomy names
            object_poses: Dict mapping taxonomy names to pose data (minimal - only 2D bbox)
            taxonomy_to_sm_names: Dict mapping taxonomy names to list of SM names
            
        Returns:
            Tuple of (filtered objects list, validated bbox mapping: taxonomy_name -> bbox_info)
        """
        import json
        import numpy as np
        from modules.qa_modules.visualization_utils import VisualizationUtils
        
        # Load seenable_obj_dict.json or use processed format
        seenable_obj_file = scene_path / "seenable_obj_dict.json"
        processed_format = (scene_path / "scene_annotations_split.json").exists()
        
        if processed_format:
            # For processed format, load scene_annotations_split.json
            annotations_file = scene_path / "scene_annotations_split.json"
            if not annotations_file.exists():
                logger.warning(f"No scene_annotations_split.json found, skipping filtering")
                return objects
            
            try:
                import json
                with open(annotations_file, 'r') as f:
                    annotations_data = json.load(f)
                
                # Build seenable_obj_dict from foreground objects
                foreground = annotations_data.get('foreground', {})
                seenable_obj_data = {}
                sm_to_bbox_2d = {}  # sm_name -> bbox_2d
                
                for category, category_objects in foreground.items():
                    for obj in category_objects:
                        sm_name = obj.get('object_id')
                        color = obj.get('color')
                        bbox_2d = obj.get('bbox_2d')
                        if sm_name and color and isinstance(color, list) and len(color) >= 3:
                            seenable_obj_data[sm_name] = color
                            if bbox_2d and len(bbox_2d) >= 4:
                                sm_to_bbox_2d[sm_name] = bbox_2d
                
                # Get camera dimensions from camera data
                camera_data = annotations_data.get('camera', {})
                seg_w = camera_data.get('width', 1920)
                seg_h = camera_data.get('height', 1080)
                total_pixels = float(seg_w * seg_h)
                
                logger.debug(f"Loaded {len(seenable_obj_data)} objects from processed format")
            except Exception as e:
                logger.warning(f"Could not load scene_annotations_split.json: {e}")
                return objects
        elif seenable_obj_file.exists():
            # Legacy format
            try:
                with open(seenable_obj_file, 'r') as f:
                    seenable_obj_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load seenable_obj_dict: {e}")
                return objects
            
            # Load segmentation
            seg_image_path = scene_path / "seg.png"
            if not seg_image_path.exists():
                logger.warning(f"No seg.png found, skipping segmentation filtering")
                return objects
            
            # Extract bboxes from segmentation
            all_bboxes = VisualizationUtils.extract_bboxes_from_segmentation(str(seg_image_path))
            if not all_bboxes:
                return objects
            
            seg_ids = {bbox['segment_id'] for bbox in all_bboxes}
            
            # Get segmentation image dimensions for bbox validation
            from PIL import Image
            try:
                seg_img = Image.open(seg_image_path)
                seg_w, seg_h = seg_img.size
                total_pixels = float(seg_w * seg_h)
            except Exception as e:
                logger.warning(f"Could not load segmentation image dimensions: {e}")
                seg_w = seg_h = 0
                total_pixels = 0.0
        else:
            logger.warning(f"No annotation files found, skipping filtering")
            return objects
        
        # Stricter bbox validation thresholds (matching copy_images_with_bbox_group)
        min_area_ratio = 0.005  # 0.5% of image area (stricter than median-based threshold)
        min_width_px = 40       # Minimum bbox width
        min_height_px = 40      # Minimum bbox height
        min_aspect_ratio = 0.2  # Enforce width/height and height/width >= 0.2
        
        # Objects that should not appear high in the image (rest on ground/floor)
        ground_supported = {
            'table', 'chair', 'sofa', 'stool', 'bench', 'desk', 'cabinet', 'drawer',
            'box', 'boxes', 'crate', 'toolbox', 'shelf', 'bookshelf', 'appliance'
        }
        
        # Load camera annotations for 3D projection validation (optional)
        camera_annots_data = None
        camera_annots_file = scene_path / "camera_annots.json"
        if camera_annots_file.exists():
            try:
                with open(camera_annots_file, 'r', encoding='utf-8') as f:
                    camera_annots_data = json.load(f)
            except Exception as e:
                logger.debug(f"Could not load camera annotations: {e}")
        
        # Build mapping from SM names to validated bboxes
        # For processed format: use pre-computed bbox_2d
        # For legacy format: extract from segmentation
        sm_to_bbox = {}
        
        if processed_format:
            # Process foreground objects with pre-computed bbox_2d
            for sm_name, bbox_2d in sm_to_bbox_2d.items():
                x1, y1, x2, y2 = bbox_2d[:4]
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                area = float(width * height)
                
                # Apply strict validation
                if width < min_width_px or height < min_height_px:
                    continue
                if total_pixels > 0 and (area / total_pixels) < min_area_ratio:
                    continue
                # Skip extremely skinny or flat boxes
                if height > 0 and (width / float(height)) < min_aspect_ratio:
                    continue
                if width > 0 and (height / float(width)) < min_aspect_ratio:
                    continue
                
                # Build bbox format compatible with legacy format
                color = seenable_obj_data.get(sm_name)
                if color and isinstance(color, list) and len(color) >= 3:
                    color_int = color[0] * 256 * 256 + color[1] * 256 + color[2]
                    sm_to_bbox[sm_name] = {
                        'bbox': bbox_2d,
                        'segment_id': color_int,
                        'bbox_area': area
                    }
        else:
            # Legacy format: extract from segmentation
            for bbox in all_bboxes:
                color_int = bbox['segment_id']
                x1, y1, x2, y2 = bbox['bbox'][:4]
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                area = float(width * height)
                
                # Apply strict validation (same as copy_images_with_bbox_group)
                if width < min_width_px or height < min_height_px:
                    continue
                if total_pixels > 0 and (area / total_pixels) < min_area_ratio:
                    continue
                # Skip extremely skinny or flat boxes
                if height > 0 and (width / float(height)) < min_aspect_ratio:
                    continue
                if width > 0 and (height / float(width)) < min_aspect_ratio:
                    continue
                
                # Match color to SM name
                for sm_name, color_list in seenable_obj_data.items():
                    if isinstance(color_list, list) and len(color_list) >= 3:
                        sm_color_int = color_list[0] * 256 * 256 + color_list[1] * 256 + color_list[2]
                        if sm_color_int == color_int:
                            sm_to_bbox[sm_name] = bbox
                            break
        
        # Load depth for depth filtering (optional)
        depth = None
        depth_file = scene_path / "depth.npy"
        if depth_file.exists():
            try:
                depth = np.load(depth_file)
            except Exception as e:
                logger.debug(f"Could not load depth.npy: {e}")
        
        # Calculate depth statistics if available
        object_depths = {}  # taxonomy_name -> avg_depth
        if depth is not None:
            for taxonomy_obj in objects:
                sm_names = taxonomy_to_sm_names.get(taxonomy_obj, [])
                depths = []
                for sm_name in sm_names:
                    if sm_name in sm_to_bbox:
                        bbox = sm_to_bbox[sm_name]['bbox']
                        x_min, y_min, x_max, y_max = bbox
                        x_min = max(0, min(int(x_min), depth.shape[1] - 1))
                        y_min = max(0, min(int(y_min), depth.shape[0] - 1))
                        x_max = max(0, min(int(x_max), depth.shape[1] - 1))
                        y_max = max(0, min(int(y_max), depth.shape[0] - 1))
                        
                        bbox_depth = depth[y_min:y_max+1, x_min:x_max+1]
                        if bbox_depth.size > 0:
                            valid_depths = bbox_depth[bbox_depth > 0]
                            if valid_depths.size > 0:
                                avg_depth = np.mean(valid_depths)
                                if not np.isnan(avg_depth):
                                    depths.append(avg_depth)
                
                if depths:
                    object_depths[taxonomy_obj] = np.mean(depths)
            
            # Calculate depth threshold if we have depth data
            if object_depths:
                depths_list = list(object_depths.values())
                if len(depths_list) >= 2:
                    median_depth = np.median(depths_list)
                    sorted_depths = sorted(depths_list)
                    depth_threshold = sorted_depths[-1] * 0.7
                else:
                    median_depth = None
                    depth_threshold = None
            else:
                median_depth = None
                depth_threshold = None
        else:
            median_depth = None
            depth_threshold = None
        
        # Filter objects in one pass: visibility + strict bbox validation + depth
        filtered_objects = []
        selected_sm_names = {}  # taxonomy_obj -> best SM name (largest bbox)
        for taxonomy_obj in objects:
            sm_names = taxonomy_to_sm_names.get(taxonomy_obj, [])
            best_candidate = None
            best_area = -1.0
            
            for sm_name in sm_names:
                # Check 1: Visibility in seenable_obj_data
                if sm_name not in seenable_obj_data:
                    continue
                
                color = seenable_obj_data[sm_name]
                if not isinstance(color, list) or len(color) < 3:
                    continue
                
                # For processed format, objects are already in foreground (visible)
                # For legacy format, check if color_int is in seg_ids
                if not processed_format:
                    color_int = color[0] * 256 * 256 + color[1] * 256 + color[2]
                    if color_int not in seg_ids:
                        continue
                
                # Check 2: Has validated bbox (already passed strict checks in sm_to_bbox mapping)
                if sm_name not in sm_to_bbox:
                    continue
                
                bbox = sm_to_bbox[sm_name]
                x1, y1, x2, y2 = bbox['bbox'][:4]
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                area = float(width * height)
                
                # Check 3: Vertical position sanity (ground-supported objects should not be very high)
                if seg_h > 0 and taxonomy_obj.lower() in ground_supported:
                    y_center = (y1 + y2) / 2.0
                    if (y_center / float(seg_h)) < 0.35:
                        # Likely hallucinated/incorrect association (e.g., table on ceiling)
                        logger.debug(f"Skipping {taxonomy_obj} - ground object appears too high in image (y_center={y_center:.1f}/{seg_h})")
                        continue
                
                # Check 4: Optional IoU validation with 3D projection (if available)
                use_candidate = True
                if camera_annots_data and object_poses and taxonomy_obj in object_poses:
                    try:
                        from modules.qa_modules.bbox3d_utils import compute_2d_bbox_from_3d_pose
                        proj_bbox, visible = compute_2d_bbox_from_3d_pose(object_poses[taxonomy_obj], camera_annots_data)
                        if proj_bbox and visible:
                            # Compute IoU between segmentation bbox and projected bbox
                            def _iou(b1, b2):
                                x1 = max(b1[0], b2[0])
                                y1 = max(b1[1], b2[1])
                                x2 = min(b1[2], b2[2])
                                y2 = min(b1[3], b2[3])
                                iw = max(0, x2 - x1)
                                ih = max(0, y2 - y1)
                                inter = iw * ih
                                a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
                                a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
                                denom = a1 + a2 - inter
                                return (inter / denom) if denom > 0 else 0.0
                            iou = _iou(bbox['bbox'], proj_bbox)
                            # Discard if IoU too low (likely mismatch like "table on ceiling")
                            if iou < 0.05:
                                use_candidate = False
                                logger.debug(f"Skipping {taxonomy_obj} - low IoU {iou:.3f} with 3D projection")
                    except Exception:
                        # If projection fails, keep candidate
                        use_candidate = True
                
                # Track best candidate (largest area among valid SM instances)
                if use_candidate and area > best_area:
                    best_area = area
                    best_candidate = sm_name
            
            # Check 5: Depth (if available) - only check if we have a valid bbox candidate
            if best_candidate is not None:
                if depth is not None and taxonomy_obj in object_depths:
                    avg_depth = object_depths[taxonomy_obj]
                    # Only apply depth filtering if thresholds are valid (not None)
                    if depth_threshold is not None and avg_depth > depth_threshold:
                        logger.debug(f"Filtering out background/occluded: {taxonomy_obj} (depth={avg_depth:.2f} > threshold={depth_threshold:.2f})")
                        continue
                    if median_depth is not None and avg_depth > median_depth * 1.5:
                        logger.debug(f"Filtering out background/occluded: {taxonomy_obj} (depth={avg_depth:.2f} > median*1.5={median_depth*1.5:.2f})")
                        continue
                
                # Object passed all validation checks
                filtered_objects.append(taxonomy_obj)
                # Store the selected SM name (largest bbox) for this taxonomy object
                selected_sm_names[taxonomy_obj] = best_candidate
        
        # Build validated bbox mapping: taxonomy_name -> bbox_info
        # Use the SM name that was selected during filtering (largest bbox)
        validated_bboxes = {}  # taxonomy_name -> bbox_info
        for taxonomy_obj in filtered_objects:
            selected_sm = selected_sm_names.get(taxonomy_obj)
            if selected_sm and selected_sm in sm_to_bbox:
                # Use the validated bbox from the selected SM name (largest bbox)
                validated_bboxes[taxonomy_obj] = sm_to_bbox[selected_sm]
            else:
                # Fallback: use first valid SM name's bbox if selected_sm not found
                sm_names = taxonomy_to_sm_names.get(taxonomy_obj, [])
                for sm_name in sm_names:
                    if sm_name in sm_to_bbox:
                        validated_bboxes[taxonomy_obj] = sm_to_bbox[sm_name]
                        break
        
        if len(filtered_objects) == 0:
            logger.debug(f"Filtering removed all objects for {scene_path.name}, returning original")
            return objects, {}
        
        num_filtered = len(objects) - len(filtered_objects)
        logger.info(f"Unified filtering: {len(objects)} -> {len(filtered_objects)} objects (filtered: {num_filtered}, strict bbox validation applied)")
        return filtered_objects, validated_bboxes
    
    def _filter_objects_by_segmentation_size_OLD(self, scene_path: Path, objects: List[str], object_poses: Dict[str, Dict]) -> List[str]:
        import numpy as np
        """
        Filter out objects with very small bounding boxes (likely noise/artifacts).
        
        Args:
            scene_path: Path to scene directory
            objects: List of object names (taxonomy names)
            object_poses: Dict mapping object names to their pose data
            
        Returns:
            List of objects with valid-sized bboxes
        """
        from modules.qa_modules.visualization_utils import VisualizationUtils
        
        seg_image_path = scene_path / "seg.png"
        if not seg_image_path.exists():
            return objects
        
        try:
            # Get all bboxes from segmentation
            all_bboxes = VisualizationUtils.extract_bboxes_from_segmentation(str(seg_image_path))
            
            if not all_bboxes:
                return objects
            
            # Calculate median bbox area to determine threshold
            bbox_areas = [bbox.get('bbox_area', 0) for bbox in all_bboxes]
            median_area = np.median(bbox_areas) if bbox_areas else 100
            
            # Filter threshold: objects must have at least 10% of median bbox area
            min_area_threshold = median_area * 0.10  # At least 10% of median size
            
            # Load mapping to get which SM names map to which taxonomy names
            from modules.qa_modules.visualization_utils import VisualizationUtils
            all_objects_file = Path(__file__).parent.parent / "sim_scene_object" / "data" / "object_list_v4.json"
            sm_to_taxonomy = {}
            
            try:
                if all_objects_file.exists():
                    with open(all_objects_file, 'r') as f:
                        all_objects_data = json.load(f)
                    for taxonomy_name, entries in all_objects_data.items():
                        for entry in entries:
                            sm_name = entry.get('object name', '')
                            if sm_name:
                                sm_to_taxonomy[sm_name] = taxonomy_name
            except Exception as e:
                logger.warning(f"Could not load all_objects mapping: {e}")
            
            # Build mapping from taxonomy name -> list of SM names
            taxonomy_to_sm = {}
            for sm_name, taxonomy_name in sm_to_taxonomy.items():
                if taxonomy_name not in taxonomy_to_sm:
                    taxonomy_to_sm[taxonomy_name] = []
                taxonomy_to_sm[taxonomy_name].append(sm_name)
            
            # Check if seenable_obj_dict exists to map colors to SM objects
            seenable_obj_file = scene_path / "seenable_obj_dict.json"
            if not seenable_obj_file.exists():
                return objects
            
            with open(seenable_obj_file, 'r') as f:
                seenable_obj_data = json.load(f)
            
            # Build mapping from SM names to bboxes
            sm_to_bbox = {}
            for bbox in all_bboxes:
                color_int = bbox['segment_id']
                for sm_name, color_list in seenable_obj_data.items():
                    if isinstance(color_list, list) and len(color_list) >= 3:
                        sm_color_int = color_list[0] * 256 * 256 + color_list[1] * 256 + color_list[2]
                        if sm_color_int == color_int:
                            sm_to_bbox[sm_name] = bbox
                            break
            
            # For each taxonomy object, check if it has a valid-sized bbox
            filtered_objects = []
            for taxonomy_obj in objects:
                sm_names = taxonomy_to_sm.get(taxonomy_obj, [])
                has_large_bbox = False
                
                for sm_name in sm_names:
                    if sm_name in sm_to_bbox:
                        bbox_area = sm_to_bbox[sm_name].get('bbox_area', 0)
                        if bbox_area >= min_area_threshold:
                            has_large_bbox = True
                            break
                
                if has_large_bbox:
                    filtered_objects.append(taxonomy_obj)
            
            # If size filtering removed everything, just skip it and return original objects
            if len(filtered_objects) == 0:
                logger.debug(f"Size filtering removed all objects for {scene_path.name}, skipping size filter")
                return objects
            
            num_filtered = len(objects) - len(filtered_objects)
            logger.info(f"Size filtering: {len(objects)} -> {len(filtered_objects)} objects (filtered out: {num_filtered}, min area: {min_area_threshold:.0f})")
            return filtered_objects
            
        except Exception as e:
            logger.warning(f"Could not filter by segmentation size: {e}")
            return objects
    
    
    def _split_objects_into_groups(self, objects: List[str], max_objects_per_group: int = 6) -> List[List[str]]:
        """Split objects into groups of max_objects_per_group for visualization.
        
        Simple even distribution: groups objects evenly across groups.
        
        Args:
            objects: List of object names to group
            max_objects_per_group: Maximum number of objects per group (default: 6)
            
        Returns:
            List of object groups, each with at most max_objects_per_group objects
        """
        groups = []
        
        # Simple even distribution
        num_objects = len(objects)
        if num_objects <= max_objects_per_group:
            return [objects]
        
        num_groups = (num_objects + max_objects_per_group - 1) // max_objects_per_group
        objects_per_group = num_objects // num_groups
        remainder = num_objects % num_groups
        
        idx = 0
        for i in range(num_groups):
            group_size = objects_per_group + (1 if i < remainder else 0)
            groups.append(objects[idx:idx + group_size])
            idx += group_size
        
        return groups
    
    def _update_question_image_paths(self, questions: List[Dict], group_image_paths: List[str]) -> List[Dict]:
        """Update question image paths to randomly assign to groups"""
        import random
        for question in questions:
            # Randomly assign to one of the group images
            question['image_path'] = random.choice(group_image_paths)
        return questions
    
    def _filter_questions_for_group(self, questions: List[Dict], group_objects: List[str], image_path: str, taxonomy_utils) -> List[Dict]:
        """Filter questions to only those relevant to objects in this group and update 'choose from' lists"""
        filtered = []
        for question in questions:
            # Get the correct answer(s) for this question
            answer = question.get('answer', '')
            choices = question.get('choices', [])
            
            # Check if any of the answer objects are in this group
            answer_objects = answer.split(',') if isinstance(answer, str) and ',' in answer else [answer]
            answer_objects = [a.strip() for a in answer_objects]
            
            # Debug: log what we're checking
            if len(filtered) == 0 and len(questions) == 1:  # Log for first question only
                logger.info(f"Checking question: answer={answer}, answer_objects={answer_objects}, group_objects={group_objects}")
            
            # If any answer object is in this group, include the question
            if any(obj in group_objects for obj in answer_objects):
                # Create a copy of the question with updated image path
                q_copy = question.copy()
                q_copy['image_path'] = image_path
                
                # ALWAYS set choices to group objects  
                q_copy['choices'] = group_objects
                
                # Also update the objects field to only include group objects
                q_copy['objects'] = group_objects
                
                # Regenerate reasoning with only group objects to avoid mentioning objects not in the image
                # Use the existing taxonomy_utils and cot_generator from the generator
                try:
                    # Create cot generator using the provided taxonomy_utils
                    from modules.qa_modules.cot_reasoning_utils import CoTReasoningGenerator
                    cot_generator = CoTReasoningGenerator(taxonomy_utils=taxonomy_utils)
                    
                    # Generate new reasoning with only group objects
                    new_reasoning = cot_generator.generate_comprehensive_reasoning(
                        q_copy['question_type'], 
                        q_copy['answer'], 
                        group_objects,  # Use group objects instead of full list
                        q_copy['answer']
                    )
                    q_copy['reasoning'] = new_reasoning
                except Exception as e:
                    logger.warning(f"Could not regenerate reasoning for {question.get('question', '')[:50]}: {e}")
                
                filtered.append(q_copy)
        
        return filtered
    
    def _save_results(
        self,
        questions: List[Dict],
        scene_statistics: List[Dict],
        processed_scenes: int,
        refresh_reasoning_enabled: bool,
    ):
        """Save the generated QA data"""
        # Add dual format for questions
        for idx, question in enumerate(questions):
            question['question_index'] = idx
            
            # Add question_category while keeping question_type (for sim images)
            question_type = question.get('question_type', '')
            if question_type:
                question['question_category'] = get_simplified_question_type(question_type)
            
            # Add choices if missing (should already be set, but ensure it's there)
            if 'choices' not in question or not question.get('choices'):
                question['choices'] = question.get('objects', [])
            
            # Add dual format: question/answer (with colored boxes) and original_question/original_answer (with object names)
            objects = question.get('choices', question.get('objects', []))
            if objects:
                # Try to load color mapping from objects_used.json
                image_id = question.get('image_id', '')
                box_to_object = {}
                obj_to_colored_box = {}
                
                if image_id:
                    try:
                        objects_used_file = self.output_dir / "images" / image_id / "objects_used.json"
                        if objects_used_file.exists():
                            with open(objects_used_file, 'r') as f:
                                mapping_data = json.load(f)
                            box_to_object = mapping_data.get('box_mapping', {})
                            # Create reverse mapping: object -> colored box
                            obj_to_colored_box = {obj: box for box, obj in box_to_object.items()}
                    except Exception as e:
                        logger.warning(f"Failed to load color mapping for {image_id}: {e}")
                
                # Fallback: infer colors from order if mapping not available
                if not obj_to_colored_box:
                    color_names = ["red", "green", "blue", "yellow", "orange", "pink", "purple", "magenta", "cyan", "rose", "violet", "turquoise"]
                    obj_to_colored_box = {}
                    for i, obj in enumerate(objects):
                        color_name = color_names[i % len(color_names)]
                        # Capitalize first letter of each word (matches real image format)
                        color_name_cap = ' '.join(word.capitalize() for word in color_name.split())
                        obj_to_colored_box[obj] = f"{color_name_cap} box"
                    box_to_object = {colored_box: obj for obj, colored_box in obj_to_colored_box.items()}
                
                question['box_to_object'] = box_to_object
                
                # Get original question and answer
                orig_question = question.get('question', '')
                orig_answer = question.get('answer', '')
                
                # Normalize "Option objects:" to "Objects to choose from:" format for consistency
                options_suffix = ""
                question_text_only = orig_question
                if "Option objects:" in question_text_only:
                    parts = question_text_only.split("Option objects:", 1)
                    question_text_only = parts[0].strip()
                    options_suffix = "Option objects:" + parts[1] if len(parts) > 1 else ""
                elif "Objects to choose from:" in question_text_only:
                    parts = question_text_only.split("Objects to choose from:", 1)
                    question_text_only = parts[0].strip()
                    options_suffix = "Objects to choose from:" + parts[1] if len(parts) > 1 else ""
                
                # Check if this is a spatial question - only spatial questions should have object names replaced in question text
                is_spatial = question.get('question_type', '').startswith('spatial_')
                
                # Create colored-box-based question and answer
                question_with_colored_boxes = question_text_only
                answer_with_colored_boxes = orig_answer
                
                # Replace answer with colored box if it's an object name
                if orig_answer in obj_to_colored_box:
                    answer_with_colored_boxes = obj_to_colored_box[orig_answer]
                
                # For spatial questions: replace object names in question text with colored boxes
                # For non-spatial questions: keep original question text (don't replace object names)
                if is_spatial:
                    # Extract affordance/function/material text to preserve it from replacement
                    affordance_match = re.search(r'has the (affordance|function|material) of ([^?]+)', question_with_colored_boxes)
                    protected_text = []
                    if affordance_match:
                        protected_text = affordance_match.group(2).strip().split()
                    
                    # Replace all object names in question with colored boxes using word boundaries
                    for obj_name, colored_box in obj_to_colored_box.items():
                        # Skip replacement if the object name is part of protected affordance/function text
                        if obj_name in protected_text:
                            continue
                        # Use regex with word boundaries to avoid replacing substrings
                        # For spatial questions, use "object in {colored_box}" format
                        pattern = r'\b' + re.escape(obj_name) + r'\b'
                        question_with_colored_boxes = re.sub(pattern, f"object in {colored_box}", question_with_colored_boxes)
                else:
                    # For non-spatial questions, keep the original question text unchanged
                    # Object names should NOT be replaced in the question text
                    pass
                
                # Replace object names in options suffix with colored boxes (for both spatial and non-spatial)
                if options_suffix:
                    for obj_name, colored_box in obj_to_colored_box.items():
                        pattern = r'\b' + re.escape(obj_name) + r'\b'
                        options_suffix = re.sub(pattern, colored_box, options_suffix)
                    question_with_colored_boxes = f"{question_with_colored_boxes} {options_suffix}"
                elif not is_spatial and not any(colored_box in question_with_colored_boxes for colored_box in obj_to_colored_box.values()):
                    # For non-spatial questions, if no colored boxes in question, add options suffix
                    colored_box_options = ", ".join(obj_to_colored_box.values())
                    question_with_colored_boxes = f"{question_with_colored_boxes} Objects to choose from: {colored_box_options}"
                
                # Keep original question/answer with object names
                question['original_question'] = orig_question
                question['original_answer'] = orig_answer
                
                # Replace with colored-box versions for question/answer
                question['question'] = question_with_colored_boxes
                question['answer'] = answer_with_colored_boxes
        
        # Regenerate reasoning with up-to-date templates (replicates refresh pipeline)
        if refresh_reasoning_enabled:
            try:
                updated = regenerate_reasoning(
                    questions,
                    self.taxonomy_utils,
                    self.object_utils,
                    annotations_dir=self.images_dir,
                )
                logger.info(f"Regenerated reasoning for {updated} question(s) during save.")
            except Exception as reasoning_error:
                logger.warning(f"Failed to regenerate reasoning inline: {reasoning_error}")

        # Remove metadata fields that were only needed for reasoning refresh
        for q in questions:
            q.pop("bbox_metadata", None)
            q.pop("pose_metadata", None)

        # Save all questions
        questions_file = self.output_dir / "all_questions.json"
        with open(questions_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(questions)} questions to {questions_file}")
        
        # Save scene statistics
        stats_file = self.output_dir / "scene_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(scene_statistics, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved scene statistics to {stats_file}")
        
        # Calculate question type counts (detailed types)
        question_type_counts = {}
        for question in questions:
            q_type = question.get('question_type', 'unknown')
            question_type_counts[q_type] = question_type_counts.get(q_type, 0) + 1
        
        # Calculate question category counts (simplified categories)
        question_category_counts = {}
        for question in questions:
            q_category = question.get('question_category', 'unknown')
            question_category_counts[q_category] = question_category_counts.get(q_category, 0) + 1
        
        # Save generation metadata
        metadata = {
            'generation_type': 'sim_image',
            'total_questions': len(questions),
            'total_scenes': processed_scenes,
            'images_dir': str(self.images_dir),
            'output_dir': str(self.output_dir),
            'generation_timestamp': str(Path().cwd()),
            'question_types': list(set(q.get('question_type', '') for q in questions)),
            'question_type_counts': question_type_counts,
            'question_categories': list(set(q.get('question_category', '') for q in questions)),
            'question_category_counts': question_category_counts,
            'scenes_processed': [s['scene_id'] for s in scene_statistics]
        }
        
        metadata_file = self.output_dir / "generation_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved generation metadata to {metadata_file}")
        

    def _copy_images_with_bbox(self, scene_path: Path, objects: List[Dict]) -> None:
        """Copy sim images and create bounding box visualizations"""
        try:
            # Create images directory in output
            images_dir = self.output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # Create scene subdirectory
            scene_id = scene_path.name
            scene_output_dir = images_dir / scene_id
            scene_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy and resize original lit.png image using VisualizationUtils
            # Handle both processed format and legacy format
            lit_image_path = scene_path / "lit.png"
            seg_image_path = scene_path / "seg.png"
            
            # Check for processed format
            processed_format = (scene_path / "scene_annotations_split.json").exists()
            if processed_format:
                # Use bbox_visualization_all.png as fallback if lit.png doesn't exist
                if not lit_image_path.exists():
                    bbox_viz_path = scene_path / "bbox_visualization_all.png"
                    if bbox_viz_path.exists():
                        lit_image_path = bbox_viz_path
                        logger.info(f"Using bbox_visualization_all.png as original image for {scene_id}")
            
            if lit_image_path.exists():
                target_original = scene_output_dir / "original.jpg"
                if not target_original.exists():
                    VisualizationUtils.resize_and_save_image(str(lit_image_path), target_original)
                
                # Create bbox visualization
                # For processed format, use scene_annotations_split.json data
                # For legacy format, use segmentation
                target_bbox = scene_output_dir / "bbox.jpg"
                
                if processed_format:
                    # Use pre-computed bbox_2d from scene_annotations_split.json
                    try:
                        import json
                        annotations_file = scene_path / "scene_annotations_split.json"
                        if annotations_file.exists():
                            with open(annotations_file, 'r') as f:
                                annotations_data = json.load(f)
                            
                            foreground = annotations_data.get('foreground', {})
                            detected_objects = []
                            
                            # Load SM to taxonomy mapping
                            sm_to_taxonomy = self.sm_to_taxonomy
                            
                            # Extract all foreground objects (for single group, use all objects)
                            for category, category_objects in foreground.items():
                                for obj in category_objects:
                                    sm_name = obj.get('object_id')
                                    bbox_2d = obj.get('bbox_2d')
                                    if sm_name and bbox_2d and len(bbox_2d) >= 4:
                                        taxonomy_name = sm_to_taxonomy.get(sm_name, sm_name)
                                        detected_objects.append({
                                            'bbox': bbox_2d,
                                            'class_name': taxonomy_name
                                        })
                            
                            if detected_objects:
                                VisualizationUtils.draw_2d_bbox_image(str(lit_image_path), detected_objects, target_bbox)
                            else:
                                logger.info(f"No bboxes found for {scene_id}")
                    except Exception as e:
                        logger.warning(f"Error creating bbox from processed format: {e}")
                elif seg_image_path.exists():
                    # Legacy format: use segmentation
                    VisualizationUtils.create_bbox_image_from_segmentation(
                        str(lit_image_path), str(seg_image_path), target_bbox
                    )
                    # Check if bbox image was actually created (has bboxes), otherwise remove it
                    if not target_bbox.exists():
                        logger.info(f"No bboxes found, not creating bbox.jpg for {scene_id}")
                else:
                    logger.info(f"No segmentation found for {scene_id}, skipping bbox.jpg creation")
            
        except Exception as e:
            logger.error(f"Error copying images for {scene_path.name}: {e}")
    
    def _copy_scene_image(self, scene_path: Path, image_id: str, images_dir: Path) -> None:
        """Copy the original lit.png scene image to images directory as original.jpg"""
        from modules.qa_modules import VisualizationUtils
        
        try:
            lit_image_path = scene_path / "lit.png"
            
            # Check for processed format
            processed_format = (scene_path / "scene_annotations_split.json").exists()
            if processed_format:
                # Use bbox_visualization_all.png as fallback if lit.png doesn't exist
                if not lit_image_path.exists():
                    bbox_viz_path = scene_path / "bbox_visualization_all.png"
                    if bbox_viz_path.exists():
                        lit_image_path = bbox_viz_path
                        logger.debug(f"Using bbox_visualization_all.png as original image for {image_id}")
            
            if not lit_image_path.exists():
                logger.warning(f"lit.png not found for {image_id}, skipping image copy")
                return
            
            # Create output directory
            scene_output_dir = images_dir / image_id
            scene_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy and resize original lit.png image as original.jpg
            target_original = scene_output_dir / "original.jpg"
            VisualizationUtils.resize_and_save_image(str(lit_image_path), target_original)
            logger.debug(f"Copied scene image to {target_original}")
            
        except Exception as e:
            logger.error(f"Error copying scene image for {image_id}: {e}")
    
    def _create_bbox_visualization_with_utils(self, image_path: Path, output_path: Path, objects: List[Dict]) -> None:
        """Create bounding box visualization using VisualizationUtils"""
        try:
            # Convert objects to the format expected by VisualizationUtils
            detected_objects = []
            for obj in objects:
                # Get bounding box from object data
                bbox = obj.get('bbox_2d', [])
                if bbox and len(bbox) >= 4:
                    detected_objects.append({
                        'bbox': bbox,
                        'class_name': obj.get('human_name', obj.get('sm_name', 'unknown'))
                    })
            
            if detected_objects:
                # Use VisualizationUtils to create bounding box visualization
                VisualizationUtils.draw_2d_bbox_image(str(image_path), detected_objects, output_path)
            else:
                # If no bounding boxes, just copy the original image
                import shutil
                shutil.copy2(image_path, output_path)
                logger.info(f"No bounding boxes found for {image_path.name}, copied original image")
                
        except Exception as e:
            logger.error(f"Error creating bounding box visualization: {e}")
            # Fallback: copy original image
            import shutil
            shutil.copy2(image_path, output_path)

def main():
    parser = argparse.ArgumentParser(description='Generate QA benchmark for sim images')
    parser.add_argument('--images_dir', required=True, help='Directory containing sim images')
    parser.add_argument('--output_dir', required=True, help='Output directory for QA data')
    parser.add_argument('--taxonomy_dir', help='Directory containing taxonomy data')
    parser.add_argument('--max_images', type=int, help='Maximum number of images (views) to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--min_scenes', type=int, help='Minimum number of scenes required; fail if generation drops below this threshold')
    parser.add_argument('--allow_scene_drop', action='store_true', help='Allow scene count to drop below the recorded baseline')
    parser.add_argument('--skip_reasoning_refresh', action='store_true', help='Skip final reasoning regeneration step (faster)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility (MUST be set before any random operations)
    random.seed(args.seed)
    logger.info(f"Random seed set to {args.seed} for reproducibility")
    
    # Also set numpy random seed if numpy is available
    try:
        import numpy as np
        np.random.seed(args.seed)
        logger.info(f"NumPy random seed set to {args.seed}")
    except ImportError:
        pass  # NumPy not available, skip
    
    output_dir_input = Path(args.output_dir).expanduser()
    if output_dir_input.is_absolute():
        output_dir_path = output_dir_input
    else:
        output_dir_path = Path.cwd() / output_dir_input
    output_dir_path = output_dir_path.resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    baseline_scene_count = None
    if GLOBAL_BASELINE_FILE.exists():
        try:
            baseline_scene_count = json.loads(GLOBAL_BASELINE_FILE.read_text()).get("expected_scene_count")
        except Exception as guard_read_error:
            logger.warning(f"Failed to read expected scene count from {GLOBAL_BASELINE_FILE}: {guard_read_error}")
    
    if baseline_scene_count is None:
        metadata_path = output_dir_path / "generation_metadata.json"
        if metadata_path.exists():
            try:
                baseline_scene_count = json.loads(metadata_path.read_text()).get("total_scenes")
            except Exception as metadata_read_error:
                logger.warning(f"Failed to read previous scene count from {metadata_path}: {metadata_read_error}")
    
    if baseline_scene_count is not None:
        try:
            baseline_scene_count = int(baseline_scene_count)
        except Exception:
            logger.warning(f"Ignoring non-integer baseline scene count value: {baseline_scene_count}")
            baseline_scene_count = None
    
    if args.min_scenes is not None:
        if baseline_scene_count is None:
            baseline_scene_count = args.min_scenes
        else:
            baseline_scene_count = max(baseline_scene_count, args.min_scenes)
    
    if baseline_scene_count is not None:
        logger.info(f"Scene count baseline set to {baseline_scene_count}")
    else:
        logger.info("No scene count baseline detected; this run will establish one.")
    
    # Clear previous results to avoid appending
    logger.info(f"Clearing previous results in {output_dir_path}...")
    import shutil
    for item in output_dir_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    logger.info("Previous results cleared")
    
    # Initialize generator
    generator = SimImageQAGenerator(
        images_dir=args.images_dir,
        output_dir=str(output_dir_path),
        taxonomy_dir=args.taxonomy_dir,
        expected_scene_count=baseline_scene_count,
        allow_scene_drop=args.allow_scene_drop
    )
    
    # Generate QA benchmark
    results = generator.generate_qa_benchmark(
        max_images=args.max_images,
        refresh_reasoning_enabled=not args.skip_reasoning_refresh
    )
    
    logger.info("Sim image QA benchmark generation completed successfully!")

if __name__ == "__main__":
    main()