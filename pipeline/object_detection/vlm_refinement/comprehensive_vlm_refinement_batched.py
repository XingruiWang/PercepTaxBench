#!/usr/bin/env python3
"""
Comprehensive VLM refinement with batching: Process multiple objects in one API call.
Also checks and updates the full_object_description list.

NEW FEATURES:
- Smart rate limit handling with exponential backoff
- Progress checkpointing and resume capability
- API cost tracking
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import google.generativeai as genai
from PIL import Image
import time
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchedVLMRefiner:
    def __init__(self, api_keys: List[str], object_list_file: str, checkpoint_file: str = None):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.configure_gemini()
        
        self.object_list_file = Path(object_list_file)
        self.known_objects = self.load_known_objects()
        self.new_objects = set()
        
        self.api_call_count = 0
        self.consecutive_rate_limits = 0
        self.last_api_call_time = None
        
        self.checkpoint_file = Path(checkpoint_file) if checkpoint_file else None
        self.processed_images = self.load_checkpoint() if self.checkpoint_file else set()
        
    def load_checkpoint(self) -> Set[str]:
        """Load processed images from checkpoint file"""
        if self.checkpoint_file and self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                processed = set(data.get('processed_images', []))
                logger.info(f"📂 Loaded checkpoint: {len(processed)} images already processed")
                return processed
        return set()
    
    def save_checkpoint(self, image_id: str):
        """Save progress checkpoint"""
        if self.checkpoint_file:
            self.processed_images.add(image_id)
            checkpoint_data = {
                'processed_images': list(self.processed_images),
                'last_updated': datetime.now().isoformat(),
                'api_calls_made': self.api_call_count
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
    
    def load_known_objects(self) -> Set[str]:
        """Load the list of known objects from merged_objects_list.txt"""
        if self.object_list_file.exists():
            with open(self.object_list_file, 'r') as f:
                objects = set(line.strip().lower() for line in f if line.strip())
            logger.info(f"Loaded {len(objects)} known objects from {self.object_list_file}")
            return objects
        else:
            logger.warning(f"Object list file not found: {self.object_list_file}")
            return set()
    
    def add_new_object(self, object_name: str):
        """Add a new object to the known list"""
        obj_lower = object_name.lower()
        if obj_lower not in self.known_objects:
            self.new_objects.add(object_name)
            self.known_objects.add(obj_lower)
            logger.info(f"    ➕ New object discovered: '{object_name}'")
    
    def save_new_objects(self):
        """Append new objects to the merged_objects_list.txt"""
        if self.new_objects:
            with open(self.object_list_file, 'a') as f:
                for obj in sorted(self.new_objects):
                    f.write(f"{obj}\n")
            logger.info(f"\n✅ Appended {len(self.new_objects)} new objects to {self.object_list_file}")
            logger.info(f"New objects: {', '.join(sorted(self.new_objects))}")
        
    def configure_gemini(self):
        """Configure Gemini API with current key"""
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info(f"Configured Gemini with API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def rotate_api_key(self):
        """Rotate to next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_gemini()
        logger.info(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def refine_objects_batch(self, objects_data: List[Dict], available_tags: List[str], 
                            max_retries: int = 3) -> Optional[Dict[str, str]]:
        """
        Refine multiple objects in a single API call for efficiency.
        
        Args:
            objects_data: List of dicts with 'object_name', 'current_name', 'crop_path'
            available_tags: All tags available for this image
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary mapping object_name to refined_name, or None if failed
        """
        
        # Prepare images and object info
        images_to_send = []
        object_info = []
        
        for obj in objects_data:
            if os.path.exists(obj['crop_path']):
                try:
                    img = Image.open(obj['crop_path'])
                    images_to_send.append(img)
                    object_info.append({
                        'object_name': obj['object_name'],
                        'current_name': obj['current_name'],
                        'index': len(images_to_send) - 1
                    })
                except Exception as e:
                    logger.warning(f"Failed to load {obj['crop_path']}: {e}")
        
        if not images_to_send:
            return None
        
        # Build comprehensive prompt for batch processing
        tags_str = ", ".join(available_tags)
        
        prompt_parts = [f"""You are analyzing {len(images_to_send)} cropped objects from the same image. For each object, determine if there's a more specific or accurate label from the available tags.

Available tags for this image: {tags_str}

Objects to analyze:
"""]
        
        for i, obj in enumerate(object_info, 1):
            prompt_parts.append(f"{i}. Current label: \"{obj['current_name']}\" (see image {i})")
        
        prompt_parts.append(f"""

Your task for EACH object:
1. Look at the cropped image carefully
2. Identify what the object actually is
3. Choose the MOST SPECIFIC and ACCURATE label from the available tags
4. IMPORTANT: If the current label is already the most specific option, KEEP IT UNCHANGED

Rules:
- ONLY choose from the tags listed above
- Be as SPECIFIC as possible (e.g., "bronze statue" > "statue", "rifle" > "weapon", "soldier" > "person")
- PREFER compound/descriptive labels over simple ones (e.g., "bronze statue" is better than just "statue")
- Consider visual details: material, shape, function, context, uniform, etc.
- For people: distinguish "soldier" (uniform), "man", "woman", "child", "person"
- For weapons: choose specific type like "rifle", "spear", "sword" over generic "weapon"
- For objects with material descriptors: keep the material (e.g., "bronze statue", "glass bottle")

Response format: Return ONLY a JSON object like this:
{{
  "1": "refined_label_for_object_1",
  "2": "refined_label_for_object_2",
  ...
}}

Output ONLY the JSON, nothing else. Each value must be from the available tags.""")
        
        for attempt in range(max_retries):
            try:
                if self.consecutive_rate_limits >= 3:
                    wait_time = 60
                    logger.warning(f"⚠️  Too many consecutive rate limits ({self.consecutive_rate_limits}). Pausing for {wait_time}s...")
                    time.sleep(wait_time)
                    self.consecutive_rate_limits = 0
                
                content = ["\n".join(prompt_parts)] + images_to_send
                
                self.api_call_count += 1
                self.last_api_call_time = time.time()
                
                response = self.model.generate_content(content)
                response_text = response.text.strip()
                
                self.consecutive_rate_limits = 0
                
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                refinements_dict = json.loads(response_text)
                
                results = {}
                tags_lower = {tag.lower(): tag for tag in available_tags}
                
                for idx_str, refined_name in refinements_dict.items():
                    idx = int(idx_str) - 1
                    if 0 <= idx < len(object_info):
                        obj = object_info[idx]
                        refined_clean = refined_name.strip().strip('"\'')
                        refined_lower = refined_clean.lower()
                        
                        if refined_lower in tags_lower:
                            validated_name = tags_lower[refined_lower]
                            results[obj['object_name']] = validated_name
                            
                            if validated_name.lower() != obj['current_name'].lower():
                                logger.info(f"    ✓ {obj['object_name']}: '{obj['current_name']}' → '{validated_name}'")
                            else:
                                logger.info(f"    ✓ {obj['object_name']}: '{obj['current_name']}' (confirmed)")
                        else:
                            logger.warning(f"    ⚠ {obj['object_name']}: VLM returned '{refined_clean}' (not in tags), keeping '{obj['current_name']}'")
                            results[obj['object_name']] = obj['current_name']
                
                return results
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response was: {response_text[:500]}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "resource_exhausted" in str(e).lower():
                    self.consecutive_rate_limits += 1
                    wait_time = min(2 ** (attempt + 2), 30)
                    logger.warning(f"⚠️  Rate limit hit (#{self.consecutive_rate_limits}). Waiting {wait_time}s before rotating...")
                    time.sleep(wait_time)
                    self.rotate_api_key()
                    if attempt == max_retries - 1:
                        logger.error(f"❌ Max retries reached. Giving up on this batch.")
                        return None
                else:
                    logger.error(f"Error in batch refinement: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return None
        
        return None
    
    def process_annotation_file(self, annotation_file: Path, dry_run: bool = False) -> dict:
        """Process a single annotation file and refine all objects in batches"""
        
        image_id = annotation_file.stem.replace('_refined', '')
        
        if image_id in self.processed_images:
            logger.info(f"⏭️  Skipping {image_id} (already processed)")
            return {
                'total_objects': 0,
                'refined_count': 0,
                'kept_count': 0,
                'failed_count': 0,
                'new_objects_count': 0,
                'refinements': [],
                'skipped': True
            }
        
        with open(annotation_file) as f:
            data = json.load(f)
        
        parent_dir = annotation_file.parent.parent
        crop_dir = parent_dir / "object_crops" / image_id
        
        available_tags = data.get('tags', [])
        detections = data.get('detections', [])
        
        stats = {
            'total_objects': len(detections),
            'refined_count': 0,
            'kept_count': 0,
            'failed_count': 0,
            'new_objects_count': 0,
            'refinements': []
        }
        
        logger.info(f"\nProcessing {image_id}:")
        logger.info(f"  Tags ({len(available_tags)}): {', '.join(available_tags[:25])}{'...' if len(available_tags) > 25 else ''}")
        logger.info(f"  Objects: {len(detections)}")
        
        # Prepare batch data
        objects_batch = []
        detection_map = {}
        
        for detection in detections:
            obj_name = detection['object_name']
            current_class = detection.get('class_name', detection.get('label', ''))
            
            # Skip if current label is already specific and in tags (e.g., "bronze statue")
            # Only skip if it's a compound/specific label (has space or is 2+ words)
            if current_class.lower() in [tag.lower() for tag in available_tags]:
                if ' ' in current_class or len(current_class.split()) >= 2:
                    logger.info(f"  ✓ Keeping specific label: {obj_name} = '{current_class}'")
                    stats['kept_count'] += 1
                    continue
            
            # Find crop image
            obj_parts = obj_name.split('_')
            if len(obj_parts) >= 2 and obj_parts[0] == 'obj':
                obj_num = obj_parts[1].zfill(3)
                crop_pattern = f"obj_{obj_num}_*_conf*.png"
                matching_crops = list(crop_dir.glob(crop_pattern)) if crop_dir.exists() else []
                
                if matching_crops:
                    crop_path = matching_crops[0]
                    objects_batch.append({
                        'object_name': obj_name,
                        'current_name': current_class,
                        'crop_path': str(crop_path)
                    })
                    detection_map[obj_name] = detection
                else:
                    logger.warning(f"  ✗ No crop found for {obj_name}")
                    stats['failed_count'] += 1
        
        # Process in batches of max 15 objects to avoid timeouts and safety blocks
        MAX_BATCH_SIZE = 15
        if objects_batch:
            num_batches = (len(objects_batch) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
            
            if num_batches > 1:
                logger.info(f"  Processing {len(objects_batch)} objects in {num_batches} batches (max {MAX_BATCH_SIZE} per batch)...")
            else:
                logger.info(f"  Processing {len(objects_batch)} objects in 1 batch...")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * MAX_BATCH_SIZE
                end_idx = min(start_idx + MAX_BATCH_SIZE, len(objects_batch))
                batch_subset = objects_batch[start_idx:end_idx]
                
                if num_batches > 1:
                    logger.info(f"    Batch {batch_idx + 1}/{num_batches}: {len(batch_subset)} objects...")
                
                refinements = self.refine_objects_batch(batch_subset, available_tags)
                
                if refinements:
                    for obj_name, refined_name in refinements.items():
                        detection = detection_map[obj_name]
                        current_class = detection.get('class_name', detection.get('label', ''))
                        
                        if refined_name.lower() != current_class.lower():
                            # Check if new object
                            if refined_name.lower() not in self.known_objects:
                                self.add_new_object(refined_name)
                                stats['new_objects_count'] += 1
                            
                            detection['class_name'] = refined_name
                            if 'refined_label' not in detection:
                                detection['original_label'] = current_class
                            detection['refined_label'] = refined_name
                            detection['refinement_method'] = 'comprehensive_vlm_batched'
                            
                            stats['refined_count'] += 1
                            stats['refinements'].append({
                                'object': obj_name,
                                'from': current_class,
                                'to': refined_name
                            })
                        else:
                            stats['kept_count'] += 1
                else:
                    stats['failed_count'] += len(batch_subset)
                
                time.sleep(1)  # Rate limiting between batches
        
        if not dry_run and stats['refined_count'] > 0:
            output_file = annotation_file.parent / f"{image_id}_refined.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"  ✅ Saved to {output_file}")
        
        if not dry_run:
            self.save_checkpoint(image_id)
        
        logger.info(f"  💰 API calls so far: {self.api_call_count}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Batched comprehensive VLM refinement')
    parser.add_argument('--unified-output-dir', type=str, required=True,
                       help='Path to openimages_unified_output directory')
    parser.add_argument('--api-keys', type=str, required=True,
                       help='Comma-separated list of Gemini API keys')
    parser.add_argument('--object-list', type=str, required=True,
                       help='Path to merged_objects_list.txt')
    parser.add_argument('--checkpoint-file', type=str, default=None,
                       help='Path to checkpoint file for resume capability')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without saving changes')
    parser.add_argument('--image-ids', type=str, default='',
                       help='Comma-separated list of specific image IDs to process')
    
    args = parser.parse_args()
    
    api_keys = [k.strip() for k in args.api_keys.split(',')]
    refiner = BatchedVLMRefiner(api_keys, args.object_list, args.checkpoint_file)
    
    unified_dir = Path(args.unified_output_dir)
    
    # Get list of images to process
    if args.image_ids:
        image_ids = [img.strip() for img in args.image_ids.split(',')]
        annotation_files = []
        for img_id in image_ids:
            # Always use original annotation file, not refined
            original_path = unified_dir / img_id / 'annotations' / f'{img_id}.json'
            if original_path.exists():
                annotation_files.append(original_path)
            else:
                logger.warning(f"No annotation found for {img_id}")
    else:
        annotation_files = list(unified_dir.glob('*/annotations/*.json'))
        annotation_files = [f for f in annotation_files if not f.name.endswith('_refined.json')]
    
    total_stats = {
        'images_processed': 0,
        'total_objects': 0,
        'total_refined': 0,
        'total_kept': 0,
        'total_failed': 0,
        'total_new_objects': 0,
        'all_refinements': []
    }
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"BATCHED COMPREHENSIVE VLM REFINEMENT")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total images: {len(annotation_files)}")
    logger.info(f"Already processed: {len(refiner.processed_images)}")
    logger.info(f"Remaining: {len(annotation_files) - len(refiner.processed_images)}")
    logger.info(f"Known objects: {len(refiner.known_objects)}")
    logger.info(f"Dry run: {args.dry_run}")
    if args.checkpoint_file:
        logger.info(f"Checkpoint file: {args.checkpoint_file}")
    logger.info(f"{'=' * 80}\n")
    
    start_time = time.time()
    
    for i, annot_file in enumerate(annotation_files, 1):
        elapsed = time.time() - start_time
        avg_time_per_image = elapsed / max(i - 1, 1)
        remaining_images = len(annotation_files) - i
        eta_seconds = avg_time_per_image * remaining_images
        eta_minutes = eta_seconds / 60
        
        logger.info(f"\n[{i}/{len(annotation_files)}] | ETA: {eta_minutes:.1f} min | API calls: {refiner.api_call_count}")
        
        stats = refiner.process_annotation_file(annot_file, dry_run=args.dry_run)
        
        if not stats.get('skipped', False):
            total_stats['images_processed'] += 1
            total_stats['total_objects'] += stats['total_objects']
            total_stats['total_refined'] += stats['refined_count']
            total_stats['total_kept'] += stats['kept_count']
            total_stats['total_failed'] += stats['failed_count']
            total_stats['total_new_objects'] += stats['new_objects_count']
            total_stats['all_refinements'].extend(stats['refinements'])
            
            logger.info(f"  Stats: {stats['refined_count']} refined, {stats['kept_count']} kept, {stats['failed_count']} failed, {stats['new_objects_count']} new")
    
    # Save new objects to file
    if not args.dry_run:
        refiner.save_new_objects()
    
    total_time = time.time() - start_time
    total_time_minutes = total_time / 60
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"REFINEMENT COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Images processed: {total_stats['images_processed']}")
    logger.info(f"Total objects: {total_stats['total_objects']}")
    logger.info(f"Refined: {total_stats['total_refined']}")
    logger.info(f"Kept: {total_stats['total_kept']}")
    logger.info(f"Failed: {total_stats['total_failed']}")
    logger.info(f"New objects discovered: {total_stats['total_new_objects']}")
    logger.info(f"")
    logger.info(f"💰 Total API calls made: {refiner.api_call_count}")
    logger.info(f"⏱️  Total time: {total_time_minutes:.1f} minutes")
    logger.info(f"⚡ Avg time per image: {total_time/max(total_stats['images_processed'], 1):.1f} seconds")
    
    if total_stats['all_refinements']:
        logger.info(f"\nSample refinements (first 20):")
        for ref in total_stats['all_refinements'][:20]:
            logger.info(f"  {ref['object']}: '{ref['from']}' → '{ref['to']}'")
        if len(total_stats['all_refinements']) > 20:
            logger.info(f"  ... and {len(total_stats['all_refinements']) - 20} more")
    
    logger.info(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()

