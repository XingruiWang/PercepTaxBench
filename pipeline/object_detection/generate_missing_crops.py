#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def parse_bbox(bbox_str):
    """Parse bbox string to coordinates"""
    bbox_str = bbox_str.strip('[]')
    coords = [float(x) for x in bbox_str.split()]
    return coords

def crop_object_from_image(image_path, bbox, padding=10):
    """Crop object from image using bounding box with padding"""
    img = Image.open(image_path)
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.width, x2 + padding)
    y2 = min(img.height, y2 + padding)
    
    cropped = img.crop((x1, y1, x2, y2))
    return cropped

def generate_missing_crops(unified_output_dir, dry_run=True):
    """Generate missing crop files from original images and bounding boxes"""
    
    unified_dir = Path(unified_output_dir)
    all_dirs = sorted([d for d in unified_dir.iterdir() if d.is_dir()])
    
    stats = {
        'total_images': 0,
        'total_objects': 0,
        'crops_existed': 0,
        'crops_generated': 0,
        'crops_failed': 0
    }
    
    failed_details = []
    
    print(f"{'='*70}")
    print(f"GENERATING MISSING CROP FILES")
    print(f"{'='*70}")
    print(f"Mode: {'DRY RUN (no files created)' if dry_run else 'LIVE (will create files)'}")
    print(f"Checking {len(all_dirs):,} images...")
    print(f"")
    
    for img_dir in tqdm(all_dirs, desc="Processing images"):
        ann_file = img_dir / "annotations" / f"{img_dir.name}.json"
        crop_dir = img_dir / "object_crops"
        
        if not ann_file.exists():
            continue
        
        with open(ann_file) as f:
            data = json.load(f)
            detections = data.get('detections', [])
            
            if not detections:
                continue
            
            stats['total_images'] += 1
            image_path = Path(data['image_info']['file_path'])
            
            if not image_path.exists():
                for det in detections:
                    stats['crops_failed'] += 1
                continue
            
            for detection in detections:
                stats['total_objects'] += 1
                obj_name = detection.get('object_name', '')
                target_crop = crop_dir / f"{obj_name}.jpg"
                
                if target_crop.exists():
                    stats['crops_existed'] += 1
                    continue
                
                bbox_str = detection.get('xyxy', '')
                if not bbox_str:
                    stats['crops_failed'] += 1
                    failed_details.append({
                        'image_id': img_dir.name,
                        'object_name': obj_name,
                        'reason': 'no_bbox'
                    })
                    continue
                
                bbox = parse_bbox(bbox_str)
                
                if not dry_run:
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    cropped_img = crop_object_from_image(image_path, bbox)
                    cropped_img.save(target_crop, 'JPEG', quality=95)
                
                stats['crops_generated'] += 1
    
    print(f"\n{'='*70}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*70}")
    print(f"Images processed: {stats['total_images']:,}")
    print(f"Total objects: {stats['total_objects']:,}")
    print(f"")
    print(f"Crops already existed: {stats['crops_existed']:,}")
    print(f"Crops {'would be generated' if dry_run else 'generated'}: {stats['crops_generated']:,}")
    print(f"Crops failed: {stats['crops_failed']:,}")
    print(f"")
    
    if stats['crops_failed'] > 0:
        print(f"Failed crops rate: {(stats['crops_failed']/stats['total_objects']*100):.2f}%")
        print(f"")
        if failed_details:
            print(f"Sample of failed crops (first 10):")
            for item in failed_details[:10]:
                reason = item.get('reason', 'unknown')
                print(f"  • {item['image_id']}: {item['object_name']} ({reason})")
    
    print(f"{'='*70}")
    
    if stats['crops_generated'] + stats['crops_existed'] == stats['total_objects'] - stats['crops_failed']:
        print(f"\n✅ All recoverable crops {'would be' if dry_run else 'are'} available!")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate missing crop files from original images")
    parser.add_argument("--unified-output-dir", default="../../openimages_unified_output",
                        help="Path to unified output directory")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run mode (no files created)")
    parser.add_argument("--execute", action="store_true",
                        help="Execute the generation (actually create files)")
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    stats = generate_missing_crops(args.unified_output_dir, dry_run=dry_run)
    
    if dry_run:
        print(f"\n✅ Dry run complete. Run with --execute to actually generate files.")
    else:
        print(f"\n✅ Generation complete!")

