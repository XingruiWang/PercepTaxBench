#!/usr/bin/env python3
"""
Comprehensive VLM refinement: For every detected object, use Gemini to check if there's
a more specific name in the image tags by looking at the cropped object image.
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Any, Optional
import google.generativeai as genai
from PIL import Image
import time
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveVLMRefiner:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.configure_gemini()
        
    def configure_gemini(self):
        """Configure Gemini API with current key"""
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info(f"Configured Gemini with API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def rotate_api_key(self):
        """Rotate to next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_gemini()
        logger.info(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def refine_object_with_tags(self, crop_image_path: str, current_name: str, 
                               available_tags: List[str], max_retries: int = 3) -> Optional[str]:
        """
        Use VLM to determine if there's a more specific name in the tags.
        
        Args:
            crop_image_path: Path to cropped object image
            current_name: Current detected object name
            available_tags: All tags available for this image
            max_retries: Maximum number of retry attempts
            
        Returns:
            Refined object name or None if refinement failed
        """
        
        if not os.path.exists(crop_image_path):
            logger.warning(f"Crop image not found: {crop_image_path}")
            return None
        
        # Create a comprehensive prompt
        tags_str = ", ".join(available_tags)
        prompt = f"""You are looking at a cropped object from an image. The object detection system labeled this as "{current_name}".

Here are ALL the tags available for the full image: {tags_str}

Your task:
1. Look carefully at the cropped object image
2. Identify what the object actually is
3. Check if there's a MORE SPECIFIC or MORE ACCURATE label in the available tags
4. Choose the BEST label that most accurately describes this specific object

Rules:
- ONLY choose from the tags listed above
- Be as SPECIFIC as possible (e.g., "rifle" is more specific than "weapon", "spear" is more specific than "weapon")
- If the current label "{current_name}" is already the most accurate, keep it
- Consider the visual details carefully (shape, function, context)
- For people: distinguish between "soldier" (in uniform), "man", "woman", "child", "person" (generic)
- For objects: choose the most specific functional name (e.g., "rifle" not "gun", "spear" not "weapon")

Response format: Output ONLY the single best label word/phrase from the tags, nothing else. No explanations."""

        for attempt in range(max_retries):
            try:
                # Load image
                image = Image.open(crop_image_path)
                
                # Call Gemini with image
                response = self.model.generate_content([prompt, image])
                refined_name = response.text.strip()
                
                # Clean up response (remove quotes, extra whitespace, etc.)
                refined_name = refined_name.strip('"\'').strip()
                
                # Validate response - check if it's in available tags (case-insensitive)
                tags_lower = {tag.lower(): tag for tag in available_tags}
                refined_lower = refined_name.lower()
                
                if refined_lower in tags_lower:
                    # Use the original case from tags
                    validated_name = tags_lower[refined_lower]
                    if validated_name.lower() != current_name.lower():
                        logger.info(f"    ✓ Refined: '{current_name}' → '{validated_name}'")
                        return validated_name
                    else:
                        logger.info(f"    ✓ Kept: '{current_name}' (VLM confirmed)")
                        return current_name
                else:
                    # VLM returned something not in tags
                    logger.warning(f"    ⚠ VLM returned '{refined_name}' (not in tags), keeping '{current_name}'")
                    return current_name
                    
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "resource_exhausted" in str(e).lower():
                    logger.warning(f"Rate limit hit, rotating API key...")
                    self.rotate_api_key()
                    time.sleep(2)
                else:
                    logger.error(f"Error refining {crop_image_path}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return None
        
        return None
    
    def process_annotation_file(self, annotation_file: Path, dry_run: bool = False) -> dict:
        """
        Process a single annotation file and refine all objects.
        
        Args:
            annotation_file: Path to annotation JSON file
            dry_run: If True, don't save changes
            
        Returns:
            Statistics dictionary
        """
        
        with open(annotation_file) as f:
            data = json.load(f)
        
        image_id = annotation_file.stem.replace('_refined', '')
        parent_dir = annotation_file.parent.parent
        crop_dir = parent_dir / "object_crops" / image_id
        
        available_tags = data.get('tags', [])
        detections = data.get('detections', [])
        
        stats = {
            'total_objects': len(detections),
            'refined_count': 0,
            'kept_count': 0,
            'failed_count': 0,
            'refinements': []
        }
        
        logger.info(f"\nProcessing {image_id}:")
        logger.info(f"  Available tags ({len(available_tags)}): {', '.join(available_tags[:20])}{'...' if len(available_tags) > 20 else ''}")
        logger.info(f"  Detections: {len(detections)}")
        
        for detection in detections:
            obj_name = detection['object_name']
            current_class = detection.get('class_name', detection.get('label', ''))
            
            # Find the crop image
            obj_parts = obj_name.split('_')
            if len(obj_parts) >= 2 and obj_parts[0] == 'obj':
                obj_num = obj_parts[1].zfill(3)
                crop_pattern = f"obj_{obj_num}_*_conf*.png"
                matching_crops = list(crop_dir.glob(crop_pattern)) if crop_dir.exists() else []
                
                if matching_crops:
                    crop_path = matching_crops[0]
                    logger.info(f"  Checking {obj_name}: '{current_class}'")
                    
                    refined_name = self.refine_object_with_tags(
                        str(crop_path),
                        current_class,
                        available_tags
                    )
                    
                    if refined_name:
                        if refined_name.lower() != current_class.lower():
                            detection['class_name'] = refined_name
                            if 'refined_label' not in detection:
                                detection['original_label'] = current_class
                            detection['refined_label'] = refined_name
                            detection['refinement_method'] = 'comprehensive_vlm'
                            
                            stats['refined_count'] += 1
                            stats['refinements'].append({
                                'object': obj_name,
                                'from': current_class,
                                'to': refined_name
                            })
                        else:
                            stats['kept_count'] += 1
                    else:
                        stats['failed_count'] += 1
                        logger.warning(f"    ✗ Failed to refine {obj_name}")
                    
                    time.sleep(0.5)  # Rate limiting
                else:
                    logger.warning(f"  ✗ No crop found for {obj_name}")
                    stats['failed_count'] += 1
        
        # Save refined annotations
        if not dry_run and stats['refined_count'] > 0:
            output_file = annotation_file.parent / f"{image_id}_refined.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"  ✓ Saved refined annotations to {output_file}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Comprehensive VLM refinement for all detected objects')
    parser.add_argument('--unified-output-dir', type=str, required=True,
                       help='Path to openimages_unified_output directory')
    parser.add_argument('--api-keys', type=str, required=True,
                       help='Comma-separated list of Gemini API keys')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without saving changes')
    parser.add_argument('--image-ids', type=str, default='',
                       help='Comma-separated list of specific image IDs to process (optional)')
    
    args = parser.parse_args()
    
    api_keys = [k.strip() for k in args.api_keys.split(',')]
    refiner = ComprehensiveVLMRefiner(api_keys)
    
    unified_dir = Path(args.unified_output_dir)
    
    # Get list of images to process
    if args.image_ids:
        image_ids = [img.strip() for img in args.image_ids.split(',')]
        annotation_files = []
        for img_id in image_ids:
            # Check for existing refined file first, then original
            refined_path = unified_dir / img_id / 'annotations' / f'{img_id}_refined.json'
            original_path = unified_dir / img_id / 'annotations' / f'{img_id}.json'
            if refined_path.exists():
                annotation_files.append(refined_path)
            elif original_path.exists():
                annotation_files.append(original_path)
            else:
                logger.warning(f"No annotation found for {img_id}")
    else:
        # Process all images
        annotation_files = list(unified_dir.glob('*/annotations/*.json'))
        annotation_files = [f for f in annotation_files if not f.name.endswith('_refined.json')]
    
    total_stats = {
        'images_processed': 0,
        'total_objects': 0,
        'total_refined': 0,
        'total_kept': 0,
        'total_failed': 0,
        'all_refinements': []
    }
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"COMPREHENSIVE VLM REFINEMENT")
    logger.info(f"{'=' * 80}")
    logger.info(f"Processing {len(annotation_files)} images")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"{'=' * 80}\n")
    
    for i, annot_file in enumerate(annotation_files, 1):
        logger.info(f"\n[{i}/{len(annotation_files)}] Processing {annot_file.parent.parent.name}")
        
        stats = refiner.process_annotation_file(annot_file, dry_run=args.dry_run)
        
        total_stats['images_processed'] += 1
        total_stats['total_objects'] += stats['total_objects']
        total_stats['total_refined'] += stats['refined_count']
        total_stats['total_kept'] += stats['kept_count']
        total_stats['total_failed'] += stats['failed_count']
        total_stats['all_refinements'].extend(stats['refinements'])
        
        logger.info(f"  Stats: {stats['refined_count']} refined, {stats['kept_count']} kept, {stats['failed_count']} failed")
    
    # Print final summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"COMPREHENSIVE VLM REFINEMENT COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Images processed: {total_stats['images_processed']}")
    logger.info(f"Total objects: {total_stats['total_objects']}")
    logger.info(f"Refined: {total_stats['total_refined']}")
    logger.info(f"Kept: {total_stats['total_kept']}")
    logger.info(f"Failed: {total_stats['total_failed']}")
    logger.info(f"\nAll refinements:")
    for ref in total_stats['all_refinements']:
        logger.info(f"  {ref['object']}: '{ref['from']}' → '{ref['to']}'")
    logger.info(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()

