#!/usr/bin/env python3
"""
Re-refine specific detected objects using VLM by looking at the actual cropped image.
This is useful when the initial detection might be ambiguous (e.g., spear vs rifle).
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from PIL import Image
import time
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ObjectReRefiner:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("Initialized Gemini Flash for object re-refinement")
    
    def refine_with_vlm(self, crop_image_path: str, current_name: str, 
                       available_tags: List[str], max_retries: int = 3) -> Optional[str]:
        """
        Use VLM to determine the best object name from available tags.
        
        Args:
            crop_image_path: Path to the cropped object image
            current_name: Current detected object name
            available_tags: All tags available for this image
            max_retries: Number of retries on failure
            
        Returns:
            Best matching tag or None if refinement fails
        """
        if not os.path.exists(crop_image_path):
            logger.warning(f"Crop image not found: {crop_image_path}")
            return None
        
        tags_str = ", ".join(available_tags)
        prompt = f"""Look carefully at this cropped object image. 

Current detected label: "{current_name}"

Available tags from the full image: {tags_str}

Task: Identify what this specific object actually is by looking at the image.

Rules:
1. Choose the MOST SPECIFIC and ACCURATE label from the available tags
2. If the current label is already the most accurate, keep it
3. Consider visual details carefully:
   - A rifle has a long barrel and stock
   - A spear has a pointed tip on a shaft
   - A sword has a blade
   - A gun can be various firearms
4. Look at the shape, structure, and context in the image
5. Be precise - don't use generic terms when specific ones are available

Response format: Output ONLY the single best label from the available tags, nothing else."""

        for attempt in range(max_retries):
            try:
                image = Image.open(crop_image_path)
                response = self.model.generate_content([prompt, image])
                refined_name = response.text.strip().lower()
                
                if refined_name in [tag.lower() for tag in available_tags]:
                    if refined_name != current_name.lower():
                        logger.info(f"  VLM refined: '{current_name}' → '{refined_name}'")
                    else:
                        logger.info(f"  VLM confirmed: '{current_name}'")
                    return refined_name
                else:
                    logger.warning(f"  VLM returned '{refined_name}' (not in tags), keeping '{current_name}'")
                    return current_name
                    
            except Exception as e:
                logger.error(f"Error in VLM refinement (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    def re_refine_image_objects(self, image_id: str, unified_output_dir: str,
                                object_indices: List[int] = None) -> bool:
        """
        Re-refine objects in a specific image.
        
        Args:
            image_id: Image ID to process
            unified_output_dir: Path to openimages_unified_output directory
            object_indices: List of object indices to refine (0-based), or None for all
            
        Returns:
            True if successful, False otherwise
        """
        annotation_file = Path(unified_output_dir) / image_id / "annotations" / f"{image_id}_refined.json"
        
        if not annotation_file.exists():
            annotation_file = Path(unified_output_dir) / image_id / "annotations" / f"{image_id}.json"
        
        if not annotation_file.exists():
            logger.error(f"Annotation file not found for {image_id}")
            return False
        
        with open(annotation_file) as f:
            data = json.load(f)
        
        available_tags = data.get('tags', [])
        detections = data.get('detections', [])
        crop_dir = Path(unified_output_dir) / image_id / "object_crops" / image_id
        
        if not crop_dir.exists():
            logger.error(f"Crop directory not found: {crop_dir}")
            return False
        
        logger.info(f"Processing {image_id} with {len(detections)} objects")
        logger.info(f"Available tags: {', '.join(available_tags)}")
        
        refinements_made = []
        
        for idx, detection in enumerate(detections):
            if object_indices is not None and idx not in object_indices:
                continue
            
            current_name = detection['class_name']
            obj_name = detection['object_name']
            
            obj_parts = obj_name.split('_')
            if len(obj_parts) >= 2 and obj_parts[0] == 'obj':
                obj_num = obj_parts[1].zfill(3)
                crop_pattern = f"obj_{obj_num}_*_conf*.png"
                matching_crops = list(crop_dir.glob(crop_pattern))
                
                if matching_crops:
                    crop_path = matching_crops[0]
                    logger.info(f"\n[Object {idx}] {obj_name}: currently '{current_name}'")
                    logger.info(f"  Crop: {crop_path.name}")
                    
                    refined_name = self.refine_with_vlm(
                        str(crop_path),
                        current_name,
                        available_tags
                    )
                    
                    if refined_name and refined_name != current_name.lower():
                        detection['class_name'] = refined_name
                        refinements_made.append({
                            'object_index': idx,
                            'object_name': obj_name,
                            'old_name': current_name,
                            'new_name': refined_name
                        })
                else:
                    logger.warning(f"No crop found for {obj_name}")
        
        if refinements_made:
            output_file = annotation_file.parent / f"{image_id}_refined.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"REFINEMENTS MADE: {len(refinements_made)}")
            logger.info(f"{'='*80}")
            for ref in refinements_made:
                logger.info(f"  {ref['object_name']}: '{ref['old_name']}' → '{ref['new_name']}'")
            logger.info(f"\nSaved to: {output_file}")
            return True
        else:
            logger.info("\nNo refinements needed")
            return False


def main():
    parser = argparse.ArgumentParser(description='Re-refine specific objects using VLM')
    parser.add_argument('--image_id', required=True, help='Image ID to process')
    parser.add_argument('--unified_output_dir', required=True, help='Path to openimages_unified_output')
    parser.add_argument('--api_key', required=True, help='Gemini API key')
    parser.add_argument('--objects', nargs='+', type=int, 
                       help='Object indices to refine (0-based, space-separated). If not provided, refines all.')
    
    args = parser.parse_args()
    
    refiner = ObjectReRefiner(args.api_key)
    success = refiner.re_refine_image_objects(
        args.image_id,
        args.unified_output_dir,
        args.objects
    )
    
    if success:
        logger.info("\n✓ Re-refinement complete!")
    else:
        logger.info("\n✗ Re-refinement failed or no changes made")


if __name__ == "__main__":
    main()

