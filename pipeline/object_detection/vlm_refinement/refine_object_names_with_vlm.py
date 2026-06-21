#!/usr/bin/env python3
"""
Use VLM (Gemini Flash) to refine object names by looking at cropped bounding boxes.
This fixes cases where GroundingDINO uses generic terms (army, person) when more specific terms exist in tags.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from PIL import Image
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VLMObjectNameRefiner:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.configure_gemini()
        
    def configure_gemini(self):
        """Configure Gemini API with current key"""
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info(f"Configured Gemini 1.5 Flash with API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def rotate_api_key(self):
        """Rotate to next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_gemini()
        logger.info(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def refine_object_name(self, image_path: str, current_name: str, 
                          possible_alternatives: List[str], 
                          max_retries: int = 3) -> Optional[str]:
        """Use VLM to determine the best object name from alternatives"""
        
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return None
        
        # Create prompt
        alternatives_str = ", ".join(possible_alternatives)
        prompt = f"""Look at this cropped object image. The current label is "{current_name}".

Available more specific labels from the image tags: {alternatives_str}

Which label is most accurate for this specific object? 

Rules:
1. Choose ONLY from the available labels listed above
2. If none fit well, respond with the current label: "{current_name}"
3. Be specific - if you see a man, say "man" not "person"
4. Consider context - soldiers in uniform should be "soldier", people in casual clothes could be "man/woman/child"

Response format: Just output the single best label word, nothing else."""

        for attempt in range(max_retries):
            try:
                # Load image
                image = Image.open(image_path)
                
                # Call Gemini
                response = self.model.generate_content([prompt, image])
                refined_name = response.text.strip().lower()
                
                # Validate response
                if refined_name in [alt.lower() for alt in possible_alternatives]:
                    logger.info(f"  Refined: '{current_name}' → '{refined_name}'")
                    return refined_name
                elif refined_name == current_name.lower():
                    logger.info(f"  Kept: '{current_name}' (VLM agreed)")
                    return current_name
                else:
                    logger.warning(f"  VLM returned unexpected: '{refined_name}', keeping '{current_name}'")
                    return current_name
                    
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logger.warning(f"Rate limit hit, rotating API key...")
                    self.rotate_api_key()
                    time.sleep(2)
                else:
                    logger.error(f"Error refining {image_path}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        return None
        
        return None
    
    def process_unified_output(self, annotation_file: Path, 
                               refinement_candidates: Dict[str, List[str]]) -> Dict[str, Any]:
        """Process a single unified output annotation file"""
        
        with open(annotation_file) as f:
            data = json.load(f)
        
        image_id = annotation_file.stem
        parent_dir = annotation_file.parent.parent
        
        available_tags = [t.lower() for t in data.get('tags', [])]
        detections = data.get('detections', [])
        
        refinements_made = []
        
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
                    # Find crop image - they're in object_crops/<image_id>/obj_###_<class>_conf*.png
                    crop_dir = parent_dir / "object_crops" / image_id
                    
                    # Extract object number from obj_name (e.g., "obj_01_army" -> "001")
                    obj_parts = obj_name.split('_')
                    if len(obj_parts) >= 2 and obj_parts[0] == 'obj':
                        obj_num = obj_parts[1].zfill(3)  # Pad to 3 digits
                        # Look for matching crop file
                        crop_pattern = f"obj_{obj_num}_*_conf*.png"
                        matching_crops = list(crop_dir.glob(crop_pattern)) if crop_dir.exists() else []
                        
                        if matching_crops:
                            crop_path = matching_crops[0]  # Use first match
                        else:
                            crop_path = None
                    else:
                        crop_path = None
                    
                    if crop_path and crop_path.exists():
                        logger.info(f"  Checking {obj_name} (currently '{class_name}')")
                        logger.info(f"    Alternatives: {', '.join(valid_alternatives)}")
                        
                        refined_name = self.refine_object_name(
                            str(crop_path),
                            class_name,
                            valid_alternatives
                        )
                        
                        if refined_name and refined_name != class_name:
                            detection['class_name'] = refined_name
                            detection['original_class_name'] = class_name
                            detection['refined_by_vlm'] = True
                            refinements_made.append({
                                'object': obj_name,
                                'original': class_name,
                                'refined': refined_name
                            })
        
        # Save updated annotation if refinements were made
        if refinements_made:
            output_file = annotation_file.parent / f"{image_id}_refined.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"  Saved refined annotation: {output_file.name}")
        
        return {
            'image_id': image_id,
            'refinements': refinements_made,
            'num_refinements': len(refinements_made)
        }
    
    def process_batch(self, unified_output_dir: str, 
                      refinement_candidates: Dict[str, List[str]],
                      max_files: int = None) -> Dict[str, Any]:
        """Process multiple annotation files"""
        
        unified_dir = Path(unified_output_dir)
        annotation_files = list(unified_dir.glob("*/annotations/*.json"))
        
        # Filter for files that don't already have _refined
        annotation_files = [f for f in annotation_files if '_refined' not in f.name and '_summary' not in f.name]
        
        if max_files:
            annotation_files = annotation_files[:max_files]
        
        logger.info(f"Processing {len(annotation_files)} files...")
        
        results = []
        total_refinements = 0
        
        for i, ann_file in enumerate(annotation_files):
            logger.info(f"[{i+1}/{len(annotation_files)}] Processing {ann_file.stem}")
            
            result = self.process_unified_output(ann_file, refinement_candidates)
            results.append(result)
            total_refinements += result['num_refinements']
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(annotation_files)} files, {total_refinements} refinements made")
        
        return {
            'total_files': len(annotation_files),
            'files_refined': len([r for r in results if r['num_refinements'] > 0]),
            'total_refinements': total_refinements,
            'results': results
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Refine object names using VLM')
    parser.add_argument('--unified_dir', type=str,
                       default='/path/to/project/openimages_unified_output',
                       help='Path to unified output directory')
    parser.add_argument('--api_keys', type=str, required=True,
                       help='Comma-separated Gemini API keys')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum files to process')
    parser.add_argument('--output', type=str,
                       default='vlm_refinement_results.json',
                       help='Output results file')
    parser.add_argument('--candidates_file', type=str,
                       default='detection_tag_analysis_refined_mappings.json',
                       help='Refinement candidates file')
    
    args = parser.parse_args()
    
    # Load API keys
    api_keys = [k.strip() for k in args.api_keys.split(',')]
    logger.info(f"Loaded {len(api_keys)} API keys")
    
    # Load refinement candidates
    with open(args.candidates_file) as f:
        candidates_data = json.load(f)
    
    refinement_candidates = candidates_data['generic_to_specific']
    logger.info(f"Loaded refinement candidates: {list(refinement_candidates.keys())}")
    
    # Initialize refiner
    refiner = VLMObjectNameRefiner(api_keys)
    
    # Process batch
    results = refiner.process_batch(
        args.unified_dir,
        refinement_candidates,
        max_files=args.max_files
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VLM REFINEMENT SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {results['total_files']}")
    print(f"Files with refinements: {results['files_refined']}")
    print(f"Total refinements made: {results['total_refinements']}")
    
    # Show examples
    print("\n" + "=" * 80)
    print("EXAMPLE REFINEMENTS")
    print("=" * 80)
    for result in results['results'][:10]:
        if result['refinements']:
            print(f"\nImage: {result['image_id']}")
            for ref in result['refinements']:
                print(f"  {ref['object']}: '{ref['original']}' → '{ref['refined']}'")


if __name__ == '__main__':
    main()

