#!/usr/bin/env python3
"""
Hybrid Object Detection: YOLOv8 for bounding boxes + Gemini for taxonomy matching
Uses YOLOv8 for accurate object detection, then matches to SM_names and taxonomy.
"""

import json
import os
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw
import google.generativeai as genai
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridObjectDetector:
    def __init__(self, yolo_model: str = "yolov8x.pt"):
        """Initialize hybrid detector with YOLOv8 and Gemini."""
        self.yolo = YOLO(yolo_model)
        
        self.api_keys = [
            ""
        ]
        self.current_key_index = 0
        self.api_key = self.api_keys[0]
        genai.configure(api_key=self.api_key)
        self.gemini = genai.GenerativeModel('gemini-2.5-flash')
        
        logger.info(f"Initialized with YOLOv8 model: {yolo_model}")
    
    def _rotate_api_key(self):
        """Rotate to the next API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=self.api_key)
        self.gemini = genai.GenerativeModel('gemini-2.5-flash')
        logger.info(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def detect_objects_yolo(self, image_path: str, confidence_threshold: float = 0.15, 
                           iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """
        Detect objects using YOLOv8 with lower confidence threshold for more detections.
        
        Returns:
            List of detections with bbox, class, confidence
        """
        results = self.yolo(image_path, conf=confidence_threshold, iou=iou_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls]
                
                detections.append({
                    'bbox': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2)
                    },
                    'class': class_name,
                    'confidence': float(conf)
                })
        
        logger.info(f"YOLOv8 detected {len(detections)} objects")
        return detections
    
    def crop_object(self, image_path: str, bbox: Dict[str, int]) -> Image.Image:
        """Crop object from image using bounding box."""
        img = Image.open(image_path)
        cropped = img.crop((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']))
        return cropped
    
    def _filter_relevant_sm_objects(self, yolo_class: str, sm_objects: List[str]) -> List[str]:
        """Filter SM objects that are semantically related to the YOLO class with prioritized scoring."""
        yolo_lower = yolo_class.lower()
        
        category_keywords = {
            'chair': (['chair', 'seat'], ['stool', 'bench']),
            'bed': (['bed'], ['mattress', 'pillow', 'blanket', 'sheet', 'drawer']),
            'table': (['table', 'desk'], ['counter', 'surface']),
            'book': (['book'], ['paper', 'magazine', 'document', 'board', 'painting', 'set']),
            'bottle': (['bottle'], ['flask', 'container', 'jar', 'goods']),
            'cup': (['cup', 'mug'], ['glass', 'tumbler']),
            'remote': (['remote', 'controller'], ['control']),
            'tv': (['tv', 'television'], ['monitor', 'screen', 'display']),
            'laptop': (['laptop', 'computer'], ['notebook']),
            'keyboard': (['keyboard'], ['keys']),
            'mouse': (['mouse'], ['pointer']),
            'cell phone': (['phone', 'mobile', 'cellphone'], []),
            'microwave': (['microwave'], ['oven']),
            'refrigerator': (['refrigerator', 'fridge'], ['freezer']),
            'sink': (['sink', 'basin'], ['washbasin']),
            'toilet': (['toilet', 'commode'], ['wc']),
            'bathtub': (['bathtub', 'tub'], ['bath']),
            'couch': (['couch', 'sofa'], ['loveseat']),
            'potted plant': (['plant'], ['pot', 'planter', 'flower', 'tree']),
            'vase': (['vase'], ['jar', 'pot']),
            'clock': (['clock'], ['watch', 'timer']),
            'teddy bear': (['teddy'], ['bear', 'toy', 'plush', 'stuffed', 'rabbit']),
            'suitcase': (['suitcase', 'luggage'], ['bag', 'case', 'box']),
            'backpack': (['backpack', 'pack'], ['bag']),
            'handbag': (['handbag', 'purse'], ['bag']),
            'umbrella': (['umbrella'], ['parasol']),
            'tie': (['tie', 'necktie'], []),
            'sports ball': (['ball'], ['sports']),
            'baseball bat': (['bat'], ['stick']),
            'skateboard': (['skateboard'], ['board']),
            'surfboard': (['surfboard'], ['board']),
            'tennis racket': (['racket', 'racquet'], []),
            'frisbee': (['frisbee'], ['disc']),
            'skis': (['ski', 'skis'], []),
            'snowboard': (['snowboard'], ['board']),
            'kite': (['kite'], []),
            'baseball glove': (['glove', 'mitt'], []),
            'bowl': (['bowl'], ['dish', 'plate']),
            'banana': (['banana'], ['fruit']),
            'apple': (['apple'], ['fruit']),
            'orange': (['orange'], ['fruit']),
            'cake': (['cake'], ['dessert']),
            'pizza': (['pizza'], ['food']),
            'donut': (['donut', 'doughnut'], []),
            'sandwich': (['sandwich'], ['food']),
            'hot dog': (['hotdog'], ['food']),
            'knife': (['knife'], ['blade', 'cutter', 'tool']),
            'fork': (['fork'], ['utensil']),
            'spoon': (['spoon'], ['utensil']),
            'scissors': (['scissors', 'shears'], []),
            'toothbrush': (['toothbrush', 'brush'], ['bath', 'goods']),
            'hair drier': (['dryer', 'drier', 'hair'], []),
            'oven': (['oven', 'stove'], ['cooker']),
            'toaster': (['toaster'], ['toast']),
            'wine glass': (['wine glass', 'glass'], ['wine', 'goblet']),
            'truck': (['truck'], ['vehicle', 'car', 'toy']),
            'car': (['car'], ['vehicle', 'auto', 'toy']),
            'bus': (['bus'], ['vehicle']),
            'train': (['train'], ['locomotive']),
            'boat': (['boat', 'ship'], ['vessel']),
            'airplane': (['airplane', 'plane'], ['aircraft']),
            'bicycle': (['bicycle', 'bike'], ['cycle']),
            'motorcycle': (['motorcycle', 'motorbike'], ['bike']),
            'traffic light': (['light'], ['signal', 'traffic']),
            'fire hydrant': (['hydrant'], ['fire']),
            'stop sign': (['sign'], ['stop']),
            'parking meter': (['meter'], ['parking']),
            'bench': (['bench'], ['seat', 'chair']),
        }
        
        if yolo_lower in category_keywords:
            primary_keywords, secondary_keywords = category_keywords[yolo_lower]
        else:
            primary_keywords = [yolo_lower]
            secondary_keywords = []
        
        scored_objects = []
        for sm_obj in sm_objects:
            sm_lower = sm_obj.lower()
            score = 0
            
            for keyword in primary_keywords:
                if keyword in sm_lower:
                    score += 10
            
            for keyword in secondary_keywords:
                if keyword in sm_lower:
                    score += 1
            
            if score > 0:
                scored_objects.append((score, sm_obj))
        
        scored_objects.sort(reverse=True, key=lambda x: x[0])
        relevant_objects = [obj for score, obj in scored_objects]
        
        return relevant_objects[:30] if relevant_objects else sm_objects[:30]
    
    def match_to_sm_and_taxonomy(self, cropped_img: Image.Image, yolo_class: str, 
                                  sm_objects: List[str], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Gemini to match detected object to SM_names and get taxonomy attributes.
        """
        relevant_sm_objects = self._filter_relevant_sm_objects(yolo_class, sm_objects)
        
        if not relevant_sm_objects:
            relevant_sm_objects = sm_objects[:50]
        
        prompt = f"""You are analyzing a cropped image of an object that was detected by YOLOv8 as "{yolo_class}".

CRITICAL CONTEXT: A professional object detector already identified this as a "{yolo_class}". 
Your job is to find the MOST SPECIFIC matching name from the SM object list below.

Most relevant SM object names for "{yolo_class}":
{chr(10).join([f'- {obj}' for obj in relevant_sm_objects])}

INSTRUCTIONS:
1. The object IS a type of "{yolo_class}" (or very similar)
2. Look at the cropped image and find the MOST SPECIFIC SM object name that matches
3. Prefer specific variations (e.g., "Old Chair" instead of generic "Chair")
4. Return ONLY the exact SM object name from the list above
5. If truly no match exists, return "unknown"

Return your answer in this JSON format:
{{
    "sm_object_name": "exact name from SM list or unknown",
    "reasoning": "brief explanation of why this is the best match"
}}

Be precise and only use names from the provided SM list."""

        response_text = None
        for attempt in range(2):
            try:
                response = self.gemini.generate_content([prompt, cropped_img])
                response_text = response.text
                break
            except Exception as e:
                logger.error(f"Gemini API call failed (attempt {attempt+1}/2): {e}")
                if attempt == 0:
                    self._rotate_api_key()
                else:
                    return {
                        'sm_object_name': 'unknown',
                        'yolo_class': yolo_class,
                        'taxonomy_attributes': {}
                    }
        
        if not response_text:
            return {
                'sm_object_name': 'unknown',
                'yolo_class': yolo_class,
                'taxonomy_attributes': {}
            }
        
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_text = response_text[json_start:json_end]
            result = json.loads(json_text)
            sm_name = result.get('sm_object_name', 'unknown')
            
            if sm_name != 'unknown' and sm_name in sm_objects:
                taxonomy_attrs = self._get_taxonomy_attributes(sm_name, taxonomy)
                return {
                    'sm_object_name': sm_name,
                    'yolo_class': yolo_class,
                    'taxonomy_attributes': taxonomy_attrs
                }
        
        return {
            'sm_object_name': 'unknown',
            'yolo_class': yolo_class,
            'taxonomy_attributes': {}
        }
    
    def _get_taxonomy_attributes(self, obj_name: str, taxonomy: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Get taxonomy attributes for an object from clustering results."""
        attributes = {}
        
        for key, clusters in taxonomy.items():
            attributes[key] = []
            for cluster_name, cluster_data in clusters.items():
                objects_in_cluster = cluster_data.get('objects', [])
                if obj_name in objects_in_cluster:
                    attributes[key].append({
                        'cluster_name': cluster_name,
                        'cluster_description': cluster_data.get('description', ''),
                        'object_in_cluster': True
                    })
        
        return attributes
    
    def detect_and_match(self, image_path: str, sm_names_file: str, 
                        taxonomy_file: str, confidence_threshold: float = 0.15,
                        iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Full pipeline: YOLOv8 detection + Gemini matching.
        """
        with open(sm_names_file, 'r') as f:
            sm_objects = [line.strip() for line in f.readlines() if line.strip()]
        
        with open(taxonomy_file, 'r') as f:
            taxonomy = json.load(f)
        
        logger.info(f"Loaded {len(sm_objects)} SM objects and taxonomy with {len(taxonomy)} keys")
        
        img = Image.open(image_path)
        image_width, image_height = img.size
        
        yolo_detections = self.detect_objects_yolo(image_path, confidence_threshold, iou_threshold)
        
        detected_objects = []
        for idx, detection in enumerate(yolo_detections):
            confidence = detection['confidence']
            yolo_class = detection['class']
            
            logger.info(f"Processing detection {idx+1}/{len(yolo_detections)}: {yolo_class} (conf={confidence:.2f})")
            
            if confidence < 0.6:
                logger.info(f"  Skipping low confidence detection (< 0.6)")
                continue
            
            cropped = self.crop_object(image_path, detection['bbox'])
            
            match_result = self.match_to_sm_and_taxonomy(
                cropped, yolo_class, sm_objects, taxonomy
            )
            
            if match_result['sm_object_name'] != 'unknown':
                detected_objects.append({
                    'object_name': match_result['sm_object_name'],
                    'yolo_class': yolo_class,
                    'bounding_box': detection['bbox'],
                    'confidence': confidence,
                    'taxonomy_attributes': match_result['taxonomy_attributes']
                })
                logger.info(f"  Matched to SM object: {match_result['sm_object_name']}")
            else:
                logger.info(f"  No SM match found")
        
        high_conf_count = sum(1 for d in yolo_detections if d['confidence'] >= 0.6)
        
        results = {
            'detected_objects': detected_objects,
            'metadata': {
                'image_path': image_path,
                'image_dimensions': {'width': image_width, 'height': image_height},
                'total_objects_detected': len(detected_objects),
                'yolo_detections': len(yolo_detections),
                'yolo_high_confidence_detections': high_conf_count,
                'confidence_threshold': 0.6,
                'sm_objects_available': len(sm_objects)
            }
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save detection results to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_file}")
    
    def draw_bounding_boxes(self, image_path: str, results: Dict[str, Any], output_image_path: str):
        """Draw bounding boxes on the image with larger labels."""
        from PIL import ImageFont
        
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        try:
            font_size = 24
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        
        for idx, obj in enumerate(results.get("detected_objects", [])):
            bbox = obj.get("bounding_box", {})
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            color = colors[idx % len(colors)]
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            label = f"{obj.get('object_name', 'Unknown')} ({obj.get('confidence', 0):.2f})"
            
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_height = text_bbox[3] - text_bbox[1]
            text_width = text_bbox[2] - text_bbox[0]
            
            label_y = max(0, y1 - text_height - 8)
            draw.rectangle([x1, label_y, x1 + text_width + 8, y1], fill=color)
            draw.text((x1 + 4, label_y + 2), label, fill='white', font=font)
        
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        image.save(output_image_path)
        logger.info(f"Visualization saved to: {output_image_path}")


def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Hybrid Object Detection (YOLOv8 + Gemini)")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--image_dir", help="Path to directory containing images")
    parser.add_argument("--output_dir", default="clustering/results/hybrid_detection_results", help="Output directory")
    parser.add_argument("--sm_names", default="SM_names.txt", help="Path to SM_names.txt")
    parser.add_argument("--taxonomy", default="clustering/results/final_taxonomy.json", help="Path to taxonomy")
    parser.add_argument("--yolo_model", default="yolov8x.pt", help="YOLOv8 model")
    parser.add_argument("--confidence", type=float, default=0.15, help="YOLO confidence threshold (lower=more detections)")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold for NMS")
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be specified")
    
    detector = HybridObjectDetector(yolo_model=args.yolo_model)
    
    if args.image:
        image_files = [args.image]
    else:
        image_files = glob.glob(os.path.join(args.image_dir, "*.png")) + \
                     glob.glob(os.path.join(args.image_dir, "*.jpg")) + \
                     glob.glob(os.path.join(args.image_dir, "*.jpeg"))
        image_files.sort()
        logger.info(f"Found {len(image_files)} images to process")
    
    for image_path in image_files:
        logger.info(f"\nProcessing: {image_path}")
        
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(args.output_dir, image_basename)
        os.makedirs(image_output_dir, exist_ok=True)
        
        json_output = os.path.join(image_output_dir, f"{image_basename}_detection.json")
        viz_output = os.path.join(image_output_dir, f"{image_basename}_bbox.png")
        
        results = detector.detect_and_match(image_path, args.sm_names, args.taxonomy, 
                                           args.confidence, args.iou)
        
        detector.save_results(results, json_output)
        detector.draw_bounding_boxes(image_path, results, viz_output)
        
        print(f"\nDetection Summary for {image_basename}:")
        print(f"YOLO detections: {results['metadata']['yolo_detections']}")
        print(f"SM matched objects: {results['metadata']['total_objects_detected']}")
        print(f"Image dimensions: {results['metadata']['image_dimensions']}")


if __name__ == "__main__":
    main()

