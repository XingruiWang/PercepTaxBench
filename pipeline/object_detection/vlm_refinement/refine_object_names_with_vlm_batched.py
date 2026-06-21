#!/usr/bin/env python3
"""
OPTIMIZED: Use VLM (Gemini Flash) to refine object names with BATCHING.
Processes all objects from one image in a SINGLE API call for 3-4x speedup.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import google.generativeai as genai
from PIL import Image
import time
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VLMObjectNameRefinerBatched:
    def __init__(self, api_keys: List[str], worker_id: int = 0):
        self.api_keys = api_keys
        self.worker_id = worker_id
        self.current_key_index = 0
        self.configure_gemini()
        
    def configure_gemini(self):
        """Configure Gemini API with current key"""
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info(f"Worker {self.worker_id}: Configured Gemini 2.5 Flash with API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def rotate_api_key(self):
        """Rotate to next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_gemini()
        logger.info(f"Worker {self.worker_id}: Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def refine_objects_batch(self, objects_to_refine: List[Dict[str, Any]], 
                            max_retries: int = 3, max_batch_size: int = 4) -> Dict[str, str]:
        """
        Refine multiple objects in a SINGLE API call.
        
        Args:
            objects_to_refine: List of dicts with keys: obj_id, image_path, current_name, alternatives
            max_batch_size: Maximum number of objects to process in one API call
            
        Returns:
            Dict mapping obj_id -> refined_name
        """
        if not objects_to_refine:
            return {}
        
        # If batch is too large, split it
        if len(objects_to_refine) > max_batch_size:
            logger.info(f"Worker {self.worker_id}:   Splitting batch of {len(objects_to_refine)} into chunks of {max_batch_size}")
            all_results = {}
            for i in range(0, len(objects_to_refine), max_batch_size):
                chunk = objects_to_refine[i:i+max_batch_size]
                chunk_results = self.refine_objects_batch(chunk, max_retries, max_batch_size)
                all_results.update(chunk_results)
            return all_results
        
        # Build a concise prompt
        prompt_parts = ["Label these objects. Choose the most specific label from the given options.\n\n"]
        
        # Add each object to the prompt
        for i, obj_info in enumerate(objects_to_refine):
            obj_id = obj_info['obj_id']
            current = obj_info['current_name']
            alternatives = obj_info['alternatives']
            
            all_options = [current] + alternatives
            options_str = ", ".join(all_options)
            prompt_parts.append(f"OBJECT_{i+1}: Choose from [{options_str}]")
        
        prompt_parts.append("\n\nReturn JSON with format: {\"OBJECT_1\": \"label\", \"OBJECT_2\": \"label\", ...}")
        
        prompt_text = "\n".join(prompt_parts)
        
        # Prepare content list: [prompt, image1, image2, ...]
        content = [prompt_text]
        for obj_info in objects_to_refine:
            if os.path.exists(obj_info['image_path']):
                content.append(Image.open(obj_info['image_path']))
        
        # Make the API call with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    content,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 500
                    }
                )
                
                # Check for safety blocks or empty response
                if not response.candidates or not response.candidates[0].content.parts:
                    logger.warning(f"Worker {self.worker_id}: Response blocked (safety/empty), keeping original names")
                    return {obj['obj_id']: obj['current_name'] for obj in objects_to_refine}
                
                response_text = response.text.strip()
                
                # Parse JSON response
                # Clean markdown code blocks if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                try:
                    results = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.warning(f"Worker {self.worker_id}: Failed to parse JSON, retrying...")
                    continue
                
                # Map results back to obj_ids
                refined_map = {}
                for i, obj_info in enumerate(objects_to_refine):
                    obj_key = f"OBJECT_{i+1}"
                    obj_id = obj_info['obj_id']
                    
                    if obj_key in results:
                        refined = results[obj_key].strip().lower()
                        
                        # Validate
                        valid_options = [obj_info['current_name'].lower()] + [a.lower() for a in obj_info['alternatives']]
                        if refined in valid_options:
                            refined_map[obj_id] = refined
                            if refined != obj_info['current_name'].lower():
                                logger.info(f"Worker {self.worker_id}:   {obj_id}: '{obj_info['current_name']}' → '{refined}'")
                        else:
                            logger.warning(f"Worker {self.worker_id}:   {obj_id}: Invalid response '{refined}', keeping '{obj_info['current_name']}'")
                            refined_map[obj_id] = obj_info['current_name']
                    else:
                        logger.warning(f"Worker {self.worker_id}:   {obj_id}: Missing in response, keeping '{obj_info['current_name']}'")
                        refined_map[obj_id] = obj_info['current_name']
                
                return refined_map
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logger.warning(f"Worker {self.worker_id}: Rate limit hit, rotating API key...")
                    self.rotate_api_key()
                    time.sleep(2)
                else:
                    logger.error(f"Worker {self.worker_id}: Error in batch refinement: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        # Return current names as fallback
                        return {obj['obj_id']: obj['current_name'] for obj in objects_to_refine}
        
        return {obj['obj_id']: obj['current_name'] for obj in objects_to_refine}
    
    def process_unified_output(self, annotation_file: Path, 
                               refinement_candidates: Dict[str, List[str]]) -> Dict[str, Any]:
        """Process a single unified output annotation file using BATCHED refinement"""
        
        with open(annotation_file) as f:
            data = json.load(f)
        
        image_id = annotation_file.stem
        parent_dir = annotation_file.parent.parent
        
        available_tags = [t.lower() for t in data.get('tags', [])]
        detections = data.get('detections', [])
        
        # Collect all objects that need refinement
        objects_to_refine = []
        detection_map = {}  # Map obj_id to detection dict
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            obj_name = detection['object_name']
            
            # Check if this object is a candidate for refinement
            if class_name in refinement_candidates:
                possible_alternatives = refinement_candidates[class_name]
                
                # Filter alternatives that exist in this image's tags
                valid_alternatives = [alt for alt in possible_alternatives 
                                     if alt.lower() in available_tags]
                
                if valid_alternatives:
                    # Find crop image
                    crop_dir = parent_dir / "object_crops" / image_id
                    
                    # Extract object number from obj_name
                    obj_parts = obj_name.split('_')
                    if len(obj_parts) >= 2 and obj_parts[0] == 'obj':
                        obj_num = obj_parts[1].zfill(3)
                        crop_pattern = f"obj_{obj_num}_*_conf*.png"
                        matching_crops = list(crop_dir.glob(crop_pattern)) if crop_dir.exists() else []
                        
                        if matching_crops:
                            crop_path = matching_crops[0]
                            
                            if crop_path.exists():
                                objects_to_refine.append({
                                    'obj_id': obj_name,
                                    'image_path': str(crop_path),
                                    'current_name': class_name,
                                    'alternatives': valid_alternatives
                                })
                                detection_map[obj_name] = detection
        
        # Process all objects in ONE API call
        refinements_made = []
        if objects_to_refine:
            logger.info(f"Worker {self.worker_id}:   Refining {len(objects_to_refine)} objects in batch...")
            
            refined_map = self.refine_objects_batch(objects_to_refine)
            
            # Update detections based on refinement results
            for obj_id, refined_name in refined_map.items():
                detection = detection_map[obj_id]
                original_name = detection['class_name'].lower()
                
                if refined_name != original_name:
                    detection['class_name'] = refined_name
                    detection['original_class_name'] = original_name
                    detection['refined_by_vlm'] = True
                    refinements_made.append({
                        'object': obj_id,
                        'original': original_name,
                        'refined': refined_name
                    })
        
        # Save updated annotation if refinements were made
        if refinements_made:
            output_file = annotation_file.parent / f"{image_id}_refined.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Worker {self.worker_id}:   Saved refined annotation: {output_file.name}")
        
        return {
            'image_id': image_id,
            'refinements': refinements_made,
            'num_refinements': len(refinements_made)
        }
    
    def process_batch(self, unified_output_dir: str, 
                      refinement_candidates: Dict[str, List[str]],
                      file_list: List[Path] = None,
                      max_files: int = None,
                      progress_file: str = None) -> Dict[str, Any]:
        """Process multiple annotation files"""
        
        unified_dir = Path(unified_output_dir)
        
        if file_list is None:
            annotation_files = list(unified_dir.glob("*/annotations/*.json"))
            # Filter for files that don't already have _refined
            annotation_files = [f for f in annotation_files if '_refined' not in f.name and '_summary' not in f.name]
        else:
            annotation_files = file_list
        
        if max_files:
            annotation_files = annotation_files[:max_files]
        
        logger.info(f"Worker {self.worker_id}: Processing {len(annotation_files)} files...")
        
        results = []
        total_refinements = 0
        
        for i, ann_file in enumerate(annotation_files):
            logger.info(f"Worker {self.worker_id}: [{i+1}/{len(annotation_files)}] Processing {ann_file.stem}")
            
            result = self.process_unified_output(ann_file, refinement_candidates)
            results.append(result)
            total_refinements += result['num_refinements']
            
            # Progress update and save
            if (i + 1) % 10 == 0:
                logger.info(f"Worker {self.worker_id}: Progress: {i+1}/{len(annotation_files)} files, {total_refinements} refinements made")
                
                # Save progress periodically
                if progress_file and (i + 1) % 50 == 0:
                    progress_data = {
                        'worker_id': self.worker_id,
                        'total_files': len(annotation_files),
                        'processed_files': i + 1,
                        'files_refined': len([r for r in results if r['num_refinements'] > 0]),
                        'total_refinements': total_refinements,
                        'last_file': str(ann_file),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f, indent=2)
                    logger.info(f"Worker {self.worker_id}: Progress saved to {progress_file}")
        
        return {
            'total_files': len(annotation_files),
            'files_refined': len([r for r in results if r['num_refinements'] > 0]),
            'total_refinements': total_refinements,
            'results': results
        }


def worker_process_function(worker_id, api_key, file_chunk_indices, all_files, unified_dir, refinement_candidates, output_prefix):
    """Worker function for parallel processing (must be at module level for pickling)"""
    file_chunk = [all_files[i] for i in file_chunk_indices]
    refiner = VLMObjectNameRefinerBatched([api_key], worker_id=worker_id)
    
    progress_file = f"{output_prefix}_worker{worker_id}_progress.json"
    results = refiner.process_batch(unified_dir, refinement_candidates, file_list=file_chunk, progress_file=progress_file)
    
    # Save worker results
    output_file = f"{output_prefix}_worker{worker_id}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Refine object names using VLM with batching')
    parser.add_argument('--unified_dir', type=str,
                       default='/path/to/project/openimages_unified_output',
                       help='Path to unified output directory')
    parser.add_argument('--api_keys', type=str, required=True,
                       help='Comma-separated Gemini API keys')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum files to process')
    parser.add_argument('--output', type=str,
                       default='vlm_refinement_batched_results.json',
                       help='Output results file')
    parser.add_argument('--candidates_file', type=str,
                       default='detection_tag_analysis_refined_mappings.json',
                       help='Refinement candidates file')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing with multiple API keys')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing refined files')
    
    args = parser.parse_args()
    
    # Load API keys
    api_keys = [k.strip() for k in args.api_keys.split(',')]
    logger.info(f"Loaded {len(api_keys)} API keys")
    
    # Load refinement candidates
    with open(args.candidates_file) as f:
        candidates_data = json.load(f)
    
    refinement_candidates = candidates_data['generic_to_specific']
    logger.info(f"Loaded refinement candidates: {list(refinement_candidates.keys())}")
    
    # Get list of files to process
    unified_dir = Path(args.unified_dir)
    annotation_files = list(unified_dir.glob("*/annotations/*.json"))
    annotation_files = [f for f in annotation_files if '_refined' not in f.name and '_summary' not in f.name]
    
    # Resume support: skip files that already have _refined version
    if args.resume:
        original_count = len(annotation_files)
        annotation_files = [f for f in annotation_files if not (f.parent / f"{f.stem}_refined.json").exists()]
        logger.info(f"Resume mode: {original_count - len(annotation_files)} files already refined, {len(annotation_files)} remaining")
    
    if args.max_files:
        annotation_files = annotation_files[:args.max_files]
    
    # Parallel processing
    if args.parallel and len(api_keys) > 1:
        import multiprocessing as mp
        
        logger.info(f"Running parallel processing with {len(api_keys)} workers...")
        
        # Split files among workers - use indices instead of actual file objects
        chunk_size = len(annotation_files) // len(api_keys)
        file_chunk_indices = []
        for i in range(len(api_keys)):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < len(api_keys) - 1 else len(annotation_files)
            file_chunk_indices.append(list(range(start_idx, end_idx)))
        
        # Prepare arguments for workers
        worker_args = []
        for i in range(len(api_keys)):
            worker_args.append((
                i,  # worker_id
                api_keys[i],  # api_key
                file_chunk_indices[i],  # file indices
                annotation_files,  # all files
                str(unified_dir),  # unified_dir
                refinement_candidates,  # refinement_candidates
                args.output.replace('.json', '')  # output_prefix
            ))
        
        # Run workers
        with mp.Pool(len(api_keys)) as pool:
            worker_results = pool.starmap(worker_process_function, worker_args)
        
        # Combine results
        results = {
            'total_files': sum(r['total_files'] for r in worker_results),
            'files_refined': sum(r['files_refined'] for r in worker_results),
            'total_refinements': sum(r['total_refinements'] for r in worker_results),
            'worker_results': worker_results
        }
    else:
        # Single worker
        refiner = VLMObjectNameRefinerBatched(api_keys, worker_id=0)
        progress_file = args.output.replace('.json', '_progress.json')
        results = refiner.process_batch(
            args.unified_dir,
            refinement_candidates,
            file_list=annotation_files,
            max_files=None,
            progress_file=progress_file
        )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VLM REFINEMENT SUMMARY (BATCHED)")
    print("=" * 80)
    print(f"Total files processed: {results['total_files']}")
    print(f"Files with refinements: {results['files_refined']}")
    print(f"Total refinements made: {results['total_refinements']}")
    print("=" * 80)


if __name__ == '__main__':
    main()

