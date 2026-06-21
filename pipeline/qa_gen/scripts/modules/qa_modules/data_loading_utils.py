import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class DataLoadingUtils:
    """Utility class for loading various data files"""
    
    def __init__(self):
        pass
    
    def load_object_descriptions(self) -> Dict[str, Any]:
        """Load object descriptions from JSON file"""
        descriptions_file = Path("/path/to/SpatialReasonerDataGen/object_description/results/object_list_final/full_object_descriptions_fully_parsed.json")
        if descriptions_file.exists():
            with open(descriptions_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Object descriptions file not found: {descriptions_file}")
            return {}
    
    def load_name_mappings(self) -> Dict[str, str]:
        """Load name mappings for object class names"""
        mappings_file = Path("/path/to/SpatialReasonerDataGen/qa_gen/object_name_mappings.json")
        if mappings_file.exists():
            with open(mappings_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Name mappings file not found: {mappings_file}")
            return {}
    
    def load_qa_space_data(self) -> Dict[str, Any]:
        """Load QA space analysis data"""
        # Use path relative to qa_gen directory
        script_dir = Path(__file__).parent.parent.parent.parent
        qa_space_file = script_dir / "scripts/analysis/results/question_answer_space_analysis.json"
        if qa_space_file.exists():
            with open(qa_space_file, 'r') as f:
                return json.load(f)
        else:
            logger.error(f"QA space analysis file not found: {qa_space_file}")
            return {}
    
    def load_sm_to_human_mapping(self) -> Dict[str, str]:
        """Load SM object name to human-readable name mapping for sim images"""
        # Try to resolve the path relative to the script location
        script_dir = Path(__file__).parent.parent
        mapping_file = script_dir / "sim_scene_object/data/object_list_v4.json"
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
                
                logger.info(f"Loaded {len(sm_to_human)} SM to human object mappings from {mapping_file}")
            except Exception as e:
                logger.warning(f"Could not load SM to human mapping from {mapping_file}: {e}")
        else:
            logger.error(f"Mapping file not found: {mapping_file.absolute()}")
        
        return sm_to_human
    
    def load_scene_object_categories(self) -> Dict[str, Dict[str, List[str]]]:
        """Load scene object categories for each scene"""
        scene_categories = {}
        scene_dir = Path("modules/sim_scene_object/data/scene_with_object")
        
        if scene_dir.exists():
            for scene_file in scene_dir.glob("*_object_category.json"):
                scene_name = scene_file.stem.replace("_object_category", "")
                try:
                    with open(scene_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    scene_categories[scene_name] = data
                except Exception as e:
                    logger.warning(f"Could not load scene categories for {scene_name}: {e}")
        
        logger.info(f"Loaded object categories for {len(scene_categories)} scenes")
        return scene_categories

    def load_image_quality_ratings(self, ratings_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
        """Load image quality ratings and normalize keys for lookup"""
        if ratings_path is None:
            ratings_path = Path("/path/to/Taxonomy/Data/SimulationMetadata/scenes/image_quality_ratings.json")
        else:
            ratings_path = Path(ratings_path)

        if not ratings_path.exists():
            logger.warning(f"Image quality ratings file not found: {ratings_path}")
            return {}

        try:
            with ratings_path.open("r", encoding="utf-8") as f:
                raw_ratings = json.load(f)
        except Exception as exc:
            logger.error(f"Failed to load image quality ratings from {ratings_path}: {exc}")
            return {}

        normalized_ratings: Dict[str, Dict[str, Any]] = {}

        def should_replace(existing: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
            existing_light = existing.get("LightingExposure")
            candidate_light = candidate.get("LightingExposure")

            if isinstance(existing_light, (int, float)) and isinstance(candidate_light, (int, float)):
                if candidate_light != existing_light:
                    return candidate_light > existing_light

            existing_timestamp = existing.get("timestamp") or ""
            candidate_timestamp = candidate.get("timestamp") or ""
            return str(candidate_timestamp) > str(existing_timestamp)

        for full_key, metrics in raw_ratings.items():
            if not isinstance(metrics, dict):
                continue

            try:
                path = Path(full_key)
            except Exception:
                continue

            if path.name.lower() != "lit.png":
                continue

            scene_name = path.parent.parent.name if path.parent.parent else None
            room_id = path.parent.name if path.parent else None
            user_dir = metrics.get("user_dir")
            if not user_dir and len(path.parents) >= 3:
                user_dir = path.parents[2].name

            if not scene_name or not room_id:
                continue

            candidate_keys = {
                f"{scene_name}/{room_id}",
                f"{scene_name}_{room_id}",
            }

            if user_dir:
                candidate_keys.add(f"{user_dir}/{scene_name}/{room_id}")

            for key in candidate_keys:
                existing = normalized_ratings.get(key)
                if existing is None or should_replace(existing, metrics):
                    normalized_ratings[key] = metrics

        logger.info(f"Loaded image quality ratings for {len(normalized_ratings)} unique sim views from {ratings_path}")
        return normalized_ratings
