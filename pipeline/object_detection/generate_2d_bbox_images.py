#!/usr/bin/env python3
import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def parse_bbox(bbox_str: str) -> List[float]:
    bbox_str = bbox_str.strip('[]')
    return [float(x) for x in bbox_str.split()]


def draw_2d_bboxes(image_path: str, detections: List[Dict], output_path: str):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    for idx, det in enumerate(detections):
        bbox_xyxy = parse_bbox(det['bbox_xyxy'])
        x1, y1, x2, y2 = [int(coord) for coord in bbox_xyxy]
        
        color = colors[idx % len(colors)]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{det['class_name']} {det['confidence']:.2f}"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), color, -1)
        
        cv2.putText(img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    return True


def process_detection_directory(detection_dir: Path, image_source_dir: str, output_dir: Path):
    annotation_file = detection_dir / 'annotations' / f'{detection_dir.name}_2d_detections.json'
    
    if not annotation_file.exists():
        print(f"Annotation file not found: {annotation_file}")
        return False
    
    with open(annotation_file, 'r') as f:
        detection_data = json.load(f)
    
    image_path = detection_data['image_info'].get('file_path', '')
    if not image_path or not os.path.exists(image_path):
        alt_image_path = f"{image_source_dir}/{detection_dir.name}.jpg"
        if os.path.exists(alt_image_path):
            image_path = alt_image_path
        else:
            print(f"Image not found for {detection_dir.name}")
            return False
    
    detections = detection_data.get('detections', [])
    if not detections:
        print(f"No detections for {detection_dir.name}")
        return False
    
    output_image_dir = output_dir / detection_dir.name / 'visualizations'
    output_image_path = output_image_dir / f'{detection_dir.name}_2d_bbox.png'
    
    return draw_2d_bboxes(image_path, detections, str(output_image_path))


def generate_detection_record(detection_dir: Path, output_dir: Path):
    annotation_file = detection_dir / 'annotations' / f'{detection_dir.name}_2d_detections.json'
    
    if not annotation_file.exists():
        return None
    
    with open(annotation_file, 'r') as f:
        detection_data = json.load(f)
    
    record = {
        'image_id': detection_dir.name,
        'image_path': detection_data['image_info'].get('file_path', ''),
        'image_size': {
            'width': detection_data['image_info'].get('width', 0),
            'height': detection_data['image_info'].get('height', 0)
        },
        'tags': detection_data.get('tags', []),
        'num_detections': len(detection_data.get('detections', [])),
        'detections': detection_data.get('detections', []),
        'bbox_visualization': f'{output_dir}/{detection_dir.name}/visualizations/{detection_dir.name}_2d_bbox.png'
    }
    
    return record


def process_all_detections(detection_dir: str, image_source_dir: str, output_dir: str):
    detection_path = Path(detection_dir)
    output_path = Path(output_dir)
    
    if not detection_path.exists():
        print(f"Detection directory does not exist: {detection_dir}")
        return
    
    image_dirs = [d for d in detection_path.iterdir() if d.is_dir()]
    print(f"Found {len(image_dirs)} detection directories")
    
    processed = 0
    failed = 0
    all_records = []
    
    for image_dir in image_dirs:
        success = process_detection_directory(image_dir, image_source_dir, output_path)
        
        if success:
            record = generate_detection_record(image_dir, output_path)
            if record:
                all_records.append(record)
            processed += 1
        else:
            failed += 1
        
        if (processed + failed) % 100 == 0:
            print(f"Processed: {processed}, Failed: {failed}")
    
    records_file = output_path / 'detection_records.json'
    with open(records_file, 'w') as f:
        json.dump(all_records, f, indent=2)
    
    summary = {
        'total_images': len(image_dirs),
        'successfully_processed': processed,
        'failed': failed,
        'total_detections': sum(r['num_detections'] for r in all_records),
        'avg_detections_per_image': sum(r['num_detections'] for r in all_records) / len(all_records) if all_records else 0
    }
    
    summary_file = output_path / 'detection_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nFinal stats:")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Avg detections/image: {summary['avg_detections_per_image']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Generate 2D bounding box visualizations')
    parser.add_argument('--detection_dir', type=str,
                        default='./results/object_detection_extracted',
                        help='Directory with extracted detection data')
    parser.add_argument('--image_source_dir', type=str,
                        default='../../openimages_train_10000',
                        help='Directory with source images')
    parser.add_argument('--output_dir', type=str,
                        default='./results/object_detection_extracted',
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print(f"Generating 2D bounding box visualizations...")
    print(f"Detection dir: {args.detection_dir}")
    print(f"Image source: {args.image_source_dir}")
    print(f"Output dir: {args.output_dir}")
    
    process_all_detections(args.detection_dir, args.image_source_dir, args.output_dir)
    print("\nVisualization generation complete!")


if __name__ == "__main__":
    main()
