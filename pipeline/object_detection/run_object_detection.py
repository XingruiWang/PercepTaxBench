#!/usr/bin/env python3
import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse


def detect_objects_in_image(image_path: str, model_path: str = None):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    height, width = img.shape[:2]
    
    detections = []
    
    return {
        'image_path': image_path,
        'image_size': {'width': width, 'height': height},
        'detections': detections
    }


def save_detection_results(image_id: str, detection_data: Dict, output_dir: str):
    output_path = Path(output_dir) / image_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    annotation_dir = output_path / 'annotations'
    annotation_dir.mkdir(exist_ok=True)
    
    annotation_file = annotation_dir / f'{image_id}_2d_detections.json'
    with open(annotation_file, 'w') as f:
        json.dump(detection_data, f, indent=2)
    
    return str(annotation_file)


def crop_detected_objects(image_path: str, detections: List[Dict], output_dir: str, image_id: str):
    if not os.path.exists(image_path):
        return
    
    img = cv2.imread(image_path)
    if img is None:
        return
    
    crops_dir = Path(output_dir) / image_id / 'object_crops'
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, det in enumerate(detections):
        bbox_str = det.get('bbox_xyxy', '')
        if not bbox_str:
            continue
        
        bbox_str = bbox_str.strip('[]')
        bbox = [int(float(x)) for x in bbox_str.split()]
        x1, y1, x2, y2 = bbox
        
        crop = img[y1:y2, x1:x2]
        
        class_name = det.get('class_name', 'unknown').replace(' ', '_')
        conf = det.get('confidence', 0.0)
        crop_file = crops_dir / f'obj_{idx:03d}_{class_name}_conf{conf:.2f}.png'
        
        cv2.imwrite(str(crop_file), crop)


def draw_bboxes_and_save(image_path: str, detections: List[Dict], output_dir: str, image_id: str):
    if not os.path.exists(image_path):
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    for idx, det in enumerate(detections):
        bbox_str = det.get('bbox_xyxy', '')
        if not bbox_str:
            continue
        
        bbox_str = bbox_str.strip('[]')
        bbox = [int(float(x)) for x in bbox_str.split()]
        x1, y1, x2, y2 = bbox
        
        color = colors[idx % len(colors)]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{det['class_name']} {det['confidence']:.2f}"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1), color, -1)
        
        cv2.putText(img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    vis_dir = Path(output_dir) / image_id / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    vis_file = vis_dir / f'{image_id}_2d_bbox.png'
    cv2.imwrite(str(vis_file), img)
    
    return str(vis_file)


def main():
    parser = argparse.ArgumentParser(description='Run object detection and generate 2D bbox outputs')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str,
                        default='./results/object_detection_output',
                        help='Output directory for detection results')
    parser.add_argument('--image_id', type=str, default=None,
                        help='Image ID (defaults to filename without extension)')
    
    args = parser.parse_args()
    
    if args.image_id is None:
        args.image_id = Path(args.image_path).stem
    
    print(f"Processing image: {args.image_path}")
    print(f"Image ID: {args.image_id}")
    
    detection_data = detect_objects_in_image(args.image_path)
    
    if detection_data is None:
        print("Failed to process image")
        return
    
    annotation_file = save_detection_results(args.image_id, detection_data, args.output_dir)
    print(f"Saved annotations: {annotation_file}")
    
    if detection_data.get('detections'):
        crop_detected_objects(args.image_path, detection_data['detections'],
                            args.output_dir, args.image_id)
        print(f"Saved object crops")
        
        vis_file = draw_bboxes_and_save(args.image_path, detection_data['detections'],
                                       args.output_dir, args.image_id)
        print(f"Saved visualization: {vis_file}")
    
    print("Detection complete!")


if __name__ == "__main__":
    main()
