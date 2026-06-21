#!/usr/bin/env python3

import json
import shutil
from pathlib import Path
from tqdm import tqdm
import re

def normalize_object_name(name):
    """Normalize object name to match crop file patterns"""
    return name.replace(' ', '_')

def find_matching_crop(crop_dir_nested, object_name):
    """Find crop file matching the object name with various patterns"""
    obj_num = object_name.split('_')[1]
    obj_num_padded = obj_num.zfill(3)
    class_name = '_'.join(object_name.split('_')[2:])
    class_name_normalized = normalize_object_name(class_name)
    
    patterns = [
        f"obj_{obj_num_padded}_{class_name}_conf*.png",
        f"obj_{obj_num_padded}_{class_name_normalized}_conf*.png",
        f"obj_{obj_num}_{class_name}_conf*.png",
        f"obj_{obj_num}_{class_name_normalized}_conf*.png",
        f"obj_{obj_num_padded}*{class_name}*.png",
        f"obj_{obj_num_padded}*{class_name_normalized}*.png",
    ]
    
    for pattern in patterns:
        matches = list(crop_dir_nested.glob(pattern))
        if matches:
            return matches[0]
    
    return None

def reorganize_crops(unified_output_dir, dry_run=True):
    """Reorganize crop files from nested structure to flat structure"""
    
    unified_dir = Path(unified_output_dir)
    all_dirs = sorted([d for d in unified_dir.iterdir() if d.is_dir()])
    
    stats = {
        'total_images': len(all_dirs),
        'total_objects': 0,
        'crops_found': 0,
        'crops_missing': 0,
        'crops_copied': 0,
        'crops_skipped': 0
    }
    
    missing_details = []
    
    print(f"{'='*70}")
    print(f"REORGANIZING CROP FILES")
    print(f"{'='*70}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will copy files)'}")
    print(f"Processing {len(all_dirs):,} images...")
    print(f"")
    
    for img_dir in tqdm(all_dirs, desc="Processing images"):
        ann_file = img_dir / "annotations" / f"{img_dir.name}.json"
        crop_dir = img_dir / "object_crops"
        crop_dir_nested = crop_dir / img_dir.name
        
        if not ann_file.exists():
            continue
        
        with open(ann_file) as f:
            data = json.load(f)
            detections = data.get('detections', [])
            
            for detection in detections:
                stats['total_objects'] += 1
                obj_name = detection.get('object_name', '')
                
                target_crop = crop_dir / f"{obj_name}.jpg"
                
                if target_crop.exists():
                    stats['crops_skipped'] += 1
                    continue
                
                if crop_dir_nested.exists():
                    source_crop = find_matching_crop(crop_dir_nested, obj_name)
                    
                    if source_crop:
                        stats['crops_found'] += 1
                        
                        if not dry_run:
                            crop_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source_crop, target_crop)
                            stats['crops_copied'] += 1
                        else:
                            stats['crops_copied'] += 1
                    else:
                        stats['crops_missing'] += 1
                        missing_details.append({
                            'image_id': img_dir.name,
                            'object_name': obj_name,
                            'class_name': detection.get('class_name', '')
                        })
                else:
                    stats['crops_missing'] += 1
                    missing_details.append({
                        'image_id': img_dir.name,
                        'object_name': obj_name,
                        'class_name': detection.get('class_name', ''),
                        'reason': 'nested_dir_missing'
                    })
    
    print(f"\n{'='*70}")
    print(f"REORGANIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total images: {stats['total_images']:,}")
    print(f"Total objects: {stats['total_objects']:,}")
    print(f"")
    print(f"Crops already in place: {stats['crops_skipped']:,}")
    print(f"Crops found and {'would be copied' if dry_run else 'copied'}: {stats['crops_copied']:,}")
    print(f"Crops missing: {stats['crops_missing']:,}")
    print(f"")
    
    if stats['crops_missing'] > 0:
        print(f"Missing crops rate: {(stats['crops_missing']/stats['total_objects']*100):.2f}%")
        print(f"")
        print(f"Sample of missing crops (first 20):")
        for item in missing_details[:20]:
            reason = item.get('reason', 'not_found_in_nested')
            print(f"  • {item['image_id']}: {item['object_name']} ({reason})")
    
    print(f"{'='*70}")
    
    return stats, missing_details

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reorganize crop files from nested to flat structure")
    parser.add_argument("--unified-output-dir", default="../../openimages_unified_output",
                        help="Path to unified output directory")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run mode (no actual changes)")
    parser.add_argument("--execute", action="store_true",
                        help="Execute the reorganization (actually copy files)")
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    stats, missing = reorganize_crops(args.unified_output_dir, dry_run=dry_run)
    
    if dry_run:
        print(f"\n✅ Dry run complete. Run with --execute to actually copy files.")
    else:
        print(f"\n✅ Reorganization complete!")

