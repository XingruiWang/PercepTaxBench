#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List
import argparse


def extract_2d_detection_data(annotation_json: Dict) -> Dict:
    image_info = annotation_json.get('image_info', {})
    tags = annotation_json.get('tags', [])
    detections = annotation_json.get('detections', [])
    
    extracted_detections = []
    for det in detections:
        detection_2d = {
            'object_name': det.get('object_name', ''),
            'class_name': det.get('class_name', ''),
            'bbox_xyxy': det.get('xyxy', ''),
            'confidence': det.get('confidence', 0.0),
            'class_id': det.get('class_id', 0),
            'box_area': det.get('box_area', 0),
            'area': det.get('area', 0)
        }
        extracted_detections.append(detection_2d)
    
    return {
        'image_info': image_info,
        'tags': tags,
        'detections': extracted_detections
    }


def copy_object_detection_files(source_dir: str, target_dir: str, image_id: str):
    source_path = Path(source_dir) / image_id
    target_path = Path(target_dir) / image_id
    
    if not source_path.exists():
        print(f"Source path does not exist: {source_path}")
        return False
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    annotation_path = source_path / 'annotations' / f'{image_id}.json'
    if annotation_path.exists():
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        
        detection_data = extract_2d_detection_data(annotation_data)
        
        (target_path / 'annotations').mkdir(exist_ok=True)
        with open(target_path / 'annotations' / f'{image_id}_2d_detections.json', 'w') as f:
            json.dump(detection_data, f, indent=2)
    
    crops_dir = source_path / 'object_crops' / image_id
    if crops_dir.exists():
        target_crops_dir = target_path / 'object_crops'
        target_crops_dir.mkdir(parents=True, exist_ok=True)
        
        for crop_file in crops_dir.glob('*.png'):
            shutil.copy2(crop_file, target_crops_dir / crop_file.name)
        
        for crop_file in crops_dir.glob('*.jpg'):
            shutil.copy2(crop_file, target_crops_dir / crop_file.name)
    
    return True


def process_all_images(source_dir: str, target_dir: str, limit: int = None):
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Source directory does not exist: {source_dir}")
        return
    
    image_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    print(f"Found {len(image_dirs)} image directories")
    
    if limit:
        image_dirs = image_dirs[:limit]
        print(f"Processing first {limit} images")
    
    processed = 0
    failed = 0
    
    for image_dir in image_dirs:
        image_id = image_dir.name
        success = copy_object_detection_files(source_dir, target_dir, image_id)
        
        if success:
            processed += 1
        else:
            failed += 1
        
        if (processed + failed) % 100 == 0:
            print(f"Processed: {processed}, Failed: {failed}")
    
    print(f"\nFinal stats:")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {processed + failed}")


def main():
    parser = argparse.ArgumentParser(description='Extract 2D object detection data from unified outputs')
    parser.add_argument('--source_dir', type=str, 
                        default='../../openimages_unified_output',
                        help='Source directory with unified outputs')
    parser.add_argument('--target_dir', type=str,
                        default='./results/object_detection_extracted',
                        help='Target directory for extracted 2D detection data')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process')
    
    args = parser.parse_args()
    
    print(f"Extracting 2D detection data...")
    print(f"Source: {args.source_dir}")
    print(f"Target: {args.target_dir}")
    
    process_all_images(args.source_dir, args.target_dir, args.limit)
    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
