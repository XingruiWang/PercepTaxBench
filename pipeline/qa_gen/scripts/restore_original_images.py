#!/usr/bin/env python3
"""
Restore original.jpg files to original resolution for realimage_final_polished.
Copies original images from openimages_train_10000 to replace resized original.jpg files.
"""

import shutil
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def restore_realimage_original_images():
    """Restore original.jpg files to original resolution"""
    realimage_final = Path("taxonomy_datagen/SpatialReasonerDataGen/qa_gen/taxonomyQABench_realimage_final_polished")
    images_dir = realimage_final / "images"
    original_images_dir = Path("/path/to/project/openimages_train_10000")
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return
    
    if not original_images_dir.exists():
        logger.error(f"Original images directory not found: {original_images_dir}")
        return
    
    logger.info(f"Restoring original.jpg files in {images_dir}")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for image_id_dir in sorted(images_dir.iterdir()):
        if not image_id_dir.is_dir():
            continue
        
        image_id = image_id_dir.name
        original_file = image_id_dir / "original.jpg"
        
        if not original_file.exists():
            logger.debug(f"Skipping {image_id}: no original.jpg")
            skipped += 1
            continue
        
        try:
            # Find source image
            source_image = original_images_dir / f"{image_id}.jpg"
            if not source_image.exists():
                logger.warning(f"Source image not found for {image_id}: {source_image}")
                skipped += 1
                continue
            
            # Get original size
            source_img = Image.open(source_image)
            original_width, original_height = source_img.size
            logger.info(f"{image_id}: Restoring to original resolution {original_width}x{original_height}")
            
            # Copy source image to replace original.jpg
            shutil.copy2(source_image, original_file)
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {processed} images...")
        
        except Exception as e:
            logger.error(f"Error processing {image_id}: {e}")
            errors += 1
    
    logger.info(f"Restoration complete: {processed} processed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    restore_realimage_original_images()

