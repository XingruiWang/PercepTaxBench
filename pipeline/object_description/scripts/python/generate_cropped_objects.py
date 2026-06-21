#!/usr/bin/env python3
"""
Object Cropping Module
Generates cropped images of detected objects from annotation data.
Based on actual pipeline outputs and file patterns.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import ast

logger = logging.getLogger(__name__)

def generate_cropped_objects(annotation_path: str, image_path: str, output_dir: str) -> bool:
    """
    Generate cropped images of detected objects from annotation data.
    
    Args:
        annotation_path: Path to the annotation JSON file
        image_path: Path to the original image
        output_dir: Directory to save the cropped objects
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load annotation data
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        
        # Load original image
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return False
        
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return False
        
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Extract detections
        detections = annotation_data.get('detections', [])
        
        if not detections:
            logger.warning("No detections found in annotation data")
            return False
        
        # Create output directory structure
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_crops_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_crops_dir, exist_ok=True)
        
        # Process each detection
        successful_crops = 0
        semantic_data = []
        
        for i, detection in enumerate(detections):
            try:
                # Extract bounding box from xyxy format
                xyxy_str = detection.get('xyxy', '')
                if isinstance(xyxy_str, str):
                    # Parse string representation of array
                    xyxy = parse_array_string(xyxy_str)
                else:
                    xyxy = xyxy_str
                
                if len(xyxy) != 4:
                    logger.warning(f"Invalid xyxy for detection {i}: {xyxy}")
                    continue
                
                x1, y1, x2, y2 = xyxy
                
                # Ensure coordinates are integers and within image bounds
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(pil_image.width, int(x2))
                y2 = min(pil_image.height, int(y2))
                
                # Skip if bbox is too small or invalid
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bbox dimensions for detection {i}: {x1}, {y1}, {x2}, {y2}")
                    continue
                
                # Crop the object
                cropped_image = pil_image.crop((x1, y1, x2, y2))
                
                # Get class name and confidence
                class_name = detection.get('class_name', f'object_{i}')
                confidence = detection.get('confidence', 0.0)
                
                # Clean class name for filename
                safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_class_name = safe_class_name.replace(' ', '_')
                
                # Generate filename based on observed pattern: obj_000_[class_name]_conf[confidence].png
                crop_filename = f"obj_{i:03d}_{safe_class_name}_conf{confidence:.2f}.png"
                crop_path = os.path.join(image_crops_dir, crop_filename)
                
                # Save the cropped image
                cropped_image.save(crop_path, 'PNG')
                
                # Create semantic data entry
                semantic_entry = {
                    "id": f"obj_{i:03d}",
                    "detection": {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_name": class_name
                    },
                    "attributes": {
                        "physical": {
                            "color": "unknown",
                            "secondary_color": "none",
                            "shape": "unknown",
                            "size_category": "unknown",
                            "dimensions_2d": {
                                "width": x2 - x1,
                                "height": y2 - y1,
                                "area": (x2 - x1) * (y2 - y1)
                            }
                        },
                        "quantitative": {
                            "aspect_ratio": (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0,
                            "bbox_area": (x2 - x1) * (y2 - y1)
                        }
                    }
                }
                
                semantic_data.append(semantic_entry)
                successful_crops += 1
                logger.debug(f"Saved cropped object: {crop_path}")
                
            except Exception as e:
                logger.warning(f"Failed to crop detection {i}: {e}")
                continue
        
        # Save semantic JSON file
        semantic_path = os.path.join(image_crops_dir, f"{image_name}_semantic.json")
        with open(semantic_path, 'w') as f:
            json.dump(semantic_data, f, indent=2)
        
        logger.info(f"Successfully generated {successful_crops} cropped objects")
        logger.info(f"Semantic data saved to: {semantic_path}")
        return successful_crops > 0
        
    except Exception as e:
        logger.error(f"Failed to generate cropped objects: {e}")
        import traceback
        logger.debug(f"Error details: {traceback.format_exc()}")
        return False

def parse_array_string(array_str: str) -> List[float]:
    """Parse string representation of numpy array to list of floats."""
    try:
        # Remove brackets and whitespace
        clean_str = array_str.strip('[]').strip()
        
        # Try comma-separated first (most common format)
        if ',' in clean_str:
            parts = [part.strip() for part in clean_str.split(',') if part.strip()]
        else:
            # Fallback to space-separated
            parts = [part.strip() for part in clean_str.split() if part.strip()]
        
        return [float(part) for part in parts]
    except Exception as e:
        logger.warning(f"Failed to parse array string '{array_str}': {e}")
        return [0.0, 0.0, 0.0, 0.0]

def generate_cropped_objects_with_mask(annotation_path: str, image_path: str, output_dir: str) -> bool:
    """
    Generate cropped objects using segmentation masks if available.
    
    Args:
        annotation_path: Path to the annotation JSON file
        image_path: Path to the original image
        output_dir: Directory to save the cropped objects
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load annotation data
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        
        # Load original image
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return False
        
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return False
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract detections
        detections = annotation_data.get('detections', [])
        
        if not detections:
            logger.warning("No detections found in annotation data")
            return False
        
        # Create output directory structure
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_crops_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_crops_dir, exist_ok=True)
        
        successful_crops = 0
        semantic_data = []
        
        for i, detection in enumerate(detections):
            try:
                # Check if segmentation mask is available
                mask_str = detection.get('mask', '')
                if mask_str and mask_str != '[]':
                    # Parse mask string (simplified - in reality this would be more complex)
                    # For now, fall back to bounding box cropping
                    logger.debug(f"Mask available for detection {i}, using bbox fallback")
                
                # Extract bounding box
                xyxy_str = detection.get('xyxy', '')
                if isinstance(xyxy_str, str):
                    xyxy = parse_array_string(xyxy_str)
                else:
                    xyxy = xyxy_str
                
                if len(xyxy) != 4:
                    continue
                
                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image_rgb.shape[1], x2)
                y2 = min(image_rgb.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                cropped_image = Image.fromarray(image_rgb[y1:y2, x1:x2])
                
                # Get class name and confidence
                class_name = detection.get('class_name', f'object_{i}')
                confidence = detection.get('confidence', 0.0)
                
                # Clean class name for filename
                safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_class_name = safe_class_name.replace(' ', '_')
                
                # Generate filename
                crop_filename = f"obj_{i:03d}_{safe_class_name}_conf{confidence:.2f}.png"
                crop_path = os.path.join(image_crops_dir, crop_filename)
                
                # Save the cropped image
                cropped_image.save(crop_path, 'PNG')
                
                # Create semantic data entry
                semantic_entry = {
                    "id": f"obj_{i:03d}",
                    "detection": {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_name": class_name
                    },
                    "attributes": {
                        "physical": {
                            "color": "unknown",
                            "secondary_color": "none",
                            "shape": "unknown",
                            "size_category": "unknown",
                            "dimensions_2d": {
                                "width": x2 - x1,
                                "height": y2 - y1,
                                "area": (x2 - x1) * (y2 - y1)
                            }
                        },
                        "quantitative": {
                            "aspect_ratio": (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0,
                            "bbox_area": (x2 - x1) * (y2 - y1)
                        }
                    }
                }
                
                semantic_data.append(semantic_entry)
                successful_crops += 1
                logger.debug(f"Saved masked cropped object: {crop_path}")
                
            except Exception as e:
                logger.warning(f"Failed to crop detection {i} with mask: {e}")
                continue
        
        # Save semantic JSON file
        semantic_path = os.path.join(image_crops_dir, f"{image_name}_semantic.json")
        with open(semantic_path, 'w') as f:
            json.dump(semantic_data, f, indent=2)
        
        logger.info(f"Successfully generated {successful_crops} masked cropped objects")
        logger.info(f"Semantic data saved to: {semantic_path}")
        return successful_crops > 0
        
    except Exception as e:
        logger.error(f"Failed to generate masked cropped objects: {e}")
        return False

def create_crop_summary(annotation_path: str, output_dir: str) -> bool:
    """
    Create a summary file listing all generated crops.
    
    Args:
        annotation_path: Path to the annotation JSON file
        output_dir: Directory containing the crops
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load annotation data
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        
        detections = annotation_data.get('detections', [])
        
        if not detections:
            return False
        
        # Create summary
        summary = []
        for i, detection in enumerate(detections):
            xyxy_str = detection.get('xyxy', '')
            if isinstance(xyxy_str, str):
                xyxy = parse_array_string(xyxy_str)
            else:
                xyxy = xyxy_str
            
            class_name = detection.get('class_name', f'object_{i}')
            confidence = detection.get('confidence', 0.0)
            
            summary.append({
                'index': i,
                'class_name': class_name,
                'bbox': xyxy,
                'confidence': confidence
            })
        
        # Save summary
        summary_path = os.path.join(output_dir, 'crop_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Crop summary saved to: {summary_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create crop summary: {e}")
        return False

if __name__ == "__main__":
    # Test the cropping function
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate cropped objects from annotation data')
    parser.add_argument('--annotation', type=str, required=True, help='Path to annotation JSON file')
    parser.add_argument('--image', type=str, required=True, help='Path to original image')
    parser.add_argument('--output', type=str, required=True, help='Output directory for crops')
    parser.add_argument('--use-masks', action='store_true', help='Use segmentation masks if available')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate cropped objects
    if args.use_masks:
        success = generate_cropped_objects_with_mask(args.annotation, args.image, args.output)
    else:
        success = generate_cropped_objects(args.annotation, args.image, args.output)
    
    if success:
        print(f"Cropped objects generated successfully in: {args.output}")
        # Create summary
        create_crop_summary(args.annotation, args.output)
    else:
        print("Failed to generate cropped objects")
