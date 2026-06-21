#!/usr/bin/env python3
"""
Copy images from simimage_final to simimage_cot at original size.
Both benchmarks have the same questions, so images should match.
"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def copy_images_to_cot():
    """Copy original.jpg and bbox.jpg from simimage_final to simimage_cot"""
    final_path = Path("taxonomy_datagen/SpatialReasonerDataGen/qa_gen/taxonomyQABench_simimage_final")
    cot_path = Path("taxonomy_datagen/SpatialReasonerDataGen/qa_gen/taxonomyQABench_simimage_cot")
    
    final_images_dir = final_path / "images"
    cot_images_dir = cot_path / "images"
    
    if not final_images_dir.exists():
        logger.error(f"Final images directory not found: {final_images_dir}")
        return
    
    if not cot_images_dir.exists():
        logger.error(f"COT images directory not found: {cot_images_dir}")
        return
    
    logger.info(f"Copying images from {final_images_dir} to {cot_images_dir}")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for image_id_dir in sorted(final_images_dir.iterdir()):
        if not image_id_dir.is_dir():
            continue
        
        image_id = image_id_dir.name
        
        try:
            final_original = image_id_dir / "original.jpg"
            final_bbox = image_id_dir / "bbox.jpg"
            
            if not final_original.exists() or not final_bbox.exists():
                logger.debug(f"Skipping {image_id}: missing images in final")
                skipped += 1
                continue
            
            # Create corresponding directory in COT
            cot_image_dir = cot_images_dir / image_id
            cot_image_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy original.jpg
            cot_original = cot_image_dir / "original.jpg"
            shutil.copy2(final_original, cot_original)
            
            # Copy bbox.jpg
            cot_bbox = cot_image_dir / "bbox.jpg"
            shutil.copy2(final_bbox, cot_bbox)
            
            processed += 1
            if processed % 100 == 0:
                logger.info(f"Processed {processed} images...")
        
        except Exception as e:
            logger.error(f"Error processing {image_id}: {e}")
            errors += 1
    
    logger.info(f"Image copy complete: {processed} processed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    copy_images_to_cot()

