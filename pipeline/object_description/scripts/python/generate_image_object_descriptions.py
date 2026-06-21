#!/usr/bin/env python3
"""
Image-Specific Object Description Generator
Generates detailed descriptions for objects detected in specific images using Gemini API.
This script processes the 3D ground truth annotations and generates descriptions for each detected object.
"""

import os
import json
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageObjectDescriptionGenerator:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-exp", sleep_sec: float = 2.0):
        """
        Initialize the image object description generator.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model name
            sleep_sec: Sleep time between API calls
        """
        self.api_key = api_key
        self.model_name = model_name
        self.sleep_sec = sleep_sec
        
        # Configure Gemini
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def generate_image_object_description(self, class_name: str, image_path: str, bbox: List[float], 
                                        confidence: float, cropped_image_path: str = "", image_context: str = "") -> Dict[str, str]:
        """
        Generate a detailed description for a specific object detected in an image.
        
        Args:
            class_name: Name of the object class
            image_path: Path to the full image file
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            confidence: Detection confidence
            cropped_image_path: Path to the cropped object image
            image_context: Additional context about the image scene
            
        Returns:
            Dictionary with structured description keys
        """
        try:
            # Format 2D detection information
            bbox_width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
            bbox_height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
            bbox_area = bbox_width * bbox_height
            
            # Create the structured prompt for image-specific object description (emphasizing specific visual appearance)
            prompt = f"""

You are tasked with describing a SPECIFIC INSTANCE of an object detected in an image, not the general class. 
Your description should be based on the visual appearance of the cropped object image and detection information.

### Task
- Input: a specific instance of "{class_name}" detected in an image
- Output: a JSON object where the class name is the key, and its value is another JSON object containing exactly **11 attributes**
- Every output must contain **all 11 keys** listed below, even if some values are empty
- Values should be specific to THIS PARTICULAR OBJECT as it appears in the image, not general descriptions
- Values should be concise natural language descriptions, not lists of bullet points
- The JSON must be syntactically valid

### Required 11 keys (describe THIS specific object instance)
1. "general_description" – description of THIS specific {class_name} instance as it appears in the image
2. "shape" – the actual shape/geometric form of THIS specific object in the image
3. "texture" – the surface textures visible on THIS specific object
4. "color" – the specific colors of THIS object as it appears in the image
5. "material" – the construction/composition materials visible on THIS specific object
6. "physical properties" - choose the most appropriate descriptions from: [heavy/light, rigid/flexible, durable/fragile, smooth/rough, movable/fixed, stable/unstable, solid/liquid/gas, solid/hollow] based on THIS object's appearance
7. "functions" – what THIS specific object can be used for or is being used for
8. "affordance" - how THIS specific object can be used, which parts can be grasped/manipulated
9. "common_elements" – other elements frequently associated with this type of object
10. "common_environmental_context" – contexts where this type of object is usually found
11. "additional_details" – any specific details, variations, or decorative aspects visible on THIS object

### Detection Context (for THIS specific instance)
- Object Class: {class_name}
- Detection Confidence: {confidence:.3f}
- 2D Bounding Box: {bbox} (width: {bbox_width:.1f}, height: {bbox_height:.1f}, area: {bbox_area:.1f})
- Full Image Path: {image_path}
- Cropped Object Path: {cropped_image_path if cropped_image_path else 'Not available'}
{f"- Scene Context: {image_context}" if image_context else ""}

### Instructions
- Focus on describing THIS SPECIFIC OBJECT as it appears in the cropped image
- Use the detection information to inform your description qualitatively (e.g., "appears large in the image", "occupies a significant portion of the frame")
- Be specific about visual characteristics you can observe from the image
- For keys 1-8, describe THIS specific instance, not the general class
- For keys 9-10, you can provide general context about this type of object
- **IMPORTANT: Do NOT include any numerical values from the detection data (confidence scores, bounding box coordinates, pixel dimensions, etc.)**
- **IMPORTANT: Use only natural language descriptions - no numbers, coordinates, or technical measurements**
- **IMPORTANT: Describe size, position, and appearance in qualitative terms (e.g., "large", "small", "centered", "off to the side", "prominent", "subtle")**

Please describe THIS specific "{class_name}" instance as it appears in the image using only natural language.
"""
            
            # Generate response with increased token limit for detailed descriptions
            response = self.model.generate_content(prompt, generation_config={
                'max_output_tokens': 2000,
                'temperature': 0.7
            })
            
            if response and response.text:
                # Parse the structured response
                structured_data = self.parse_json_response(response.text)
                if structured_data:
                    # Check if all required keys are present
                    required_keys = [
                        "general_description", "shape", "texture", "color", "material",
                        "physical_properties", "functions", "affordance", "common_elements",
                        "common_environmental_context", "additional_details"
                    ]
                    
                    if all(key in structured_data for key in required_keys):
                        logger.info(f"Generated complete description for {class_name} in {os.path.basename(image_path)}")
                        return structured_data
                    else:
                        # Complete missing keys
                        logger.info(f"Generated partial description for {class_name}, completing missing keys")
                        completed_data = self.complete_missing_keys(structured_data, class_name, confidence, bbox)
                        return completed_data
                else:
                    logger.warning(f"Failed to parse structured response for {class_name}")
                    return self.generate_fallback_description(class_name, confidence, {"xyxy": bbox})
            else:
                logger.warning(f"Empty response for {class_name}")
                return self.generate_fallback_description(class_name, confidence, {"xyxy": bbox})
                
        except Exception as e:
            logger.error(f"Error generating description for {class_name}: {e}")
            return self.generate_fallback_description(class_name, confidence, {"xyxy": bbox})
    
    def parse_json_response(self, description: str) -> Dict[str, str]:
        """Parse JSON response and return structured data with keys"""
        import re
        
        # Look for JSON content between curly braces
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, description)
        
        for json_str in json_matches:
            try:
                parsed_data = json.loads(json_str)
                
                # Extract the nested object with the category name as key (like the main script)
                if isinstance(parsed_data, dict):
                    for category_name, category_data in parsed_data.items():
                        if isinstance(category_data, dict):
                            # Check if it has the required keys
                            required_keys = [
                                "general_description", "shape", "texture", "color", "material",
                                "physical_properties", "functions", "affordance", "common_elements",
                                "common_environmental_context", "additional_details"
                            ]
                            
                            if all(key in category_data for key in required_keys):
                                return category_data
                        elif isinstance(category_data, str):
                            # If the value is a string, create a simple structure
                            return {"description": category_data}
                    
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try to extract from markdown code blocks
        if "```json" in description:
            try:
                json_start = description.find("```json") + 7
                json_end = description.find("```", json_start)
                if json_end > json_start:
                    json_content = description[json_start:json_end].strip()
                    parsed_data = json.loads(json_content)
                    
                    # Extract the nested object with the category name as key (like the main script)
                    if isinstance(parsed_data, dict):
                        for category_name, category_data in parsed_data.items():
                            if isinstance(category_data, dict):
                                required_keys = [
                                    "general_description", "shape", "texture", "color", "material",
                                    "physical_properties", "functions", "affordance", "common_elements",
                                    "common_environmental_context", "additional_details"
                                ]
                                
                                if all(key in category_data for key in required_keys):
                                    return category_data
                            elif isinstance(category_data, str):
                                return {"description": category_data}
            except json.JSONDecodeError:
                pass
        
        # If still no valid JSON, try to extract key-value pairs from the text
        try:
            # Look for patterns like "key": "value" in the text
            kv_pattern = r'"([^"]+)":\s*"([^"]*)"'
            matches = re.findall(kv_pattern, description)
            if matches:
                result = {}
                for key, value in matches:
                    if key not in ["general_description", "shape", "texture", "color", "material", 
                                 "physical_properties", "functions", "affordance", "common_elements", 
                                 "common_environmental_context", "additional_details"]:
                        continue
                    result[key] = value.strip()
                if result:
                    return result
        except Exception:
            pass
        
        # If still no valid JSON, try to extract partial JSON content
        try:
            # Look for any JSON-like structure and try to parse it
            # This handles cases where the response is partially structured
            partial_json_pattern = r'\{[^{}]*"general_description"[^{}]*\}'
            partial_matches = re.findall(partial_json_pattern, description)
            
            for partial_json in partial_matches:
                try:
                    parsed_data = json.loads(partial_json)
                    # Return whatever keys we found
                    if "general_description" in parsed_data:
                        return parsed_data
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        return {}
    
    def complete_missing_keys(self, partial_description: Dict[str, str], class_name: str, confidence: float, bbox: List[float]) -> Dict[str, str]:
        """Complete missing keys in a partial description using Gemini API"""
        required_keys = [
            "general_description", "shape", "texture", "color", "material",
            "physical_properties", "functions", "affordance", "common_elements",
            "common_environmental_context", "additional_details"
        ]
        
        # Start with the partial description
        completed_description = partial_description.copy()
        
        # Calculate bbox info for context
        bbox_width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
        bbox_height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
        bbox_area = bbox_width * bbox_height
        
        # Generate missing keys using Gemini API
        missing_keys = [key for key in required_keys if key not in completed_description]
        
        if missing_keys:
            logger.info(f"Generating {len(missing_keys)} missing keys for {class_name} using Gemini API")
            
            # Create a prompt for generating missing keys
            available_info = f"Object: {class_name}, Confidence: {confidence:.3f}, Size: {bbox_width:.1f}x{bbox_height:.1f} pixels, Area: {bbox_area:.1f} square pixels"
            existing_desc = f"Existing description: {partial_description.get('general_description', 'No general description available')}"
            
            prompt = f"""
You are completing missing attributes for an object description. Please generate ONLY the missing keys in JSON format.

Object Information:
{available_info}
{existing_desc}

Missing keys to generate: {missing_keys}

### Required format for each key:
1. "general_description" – short summary of what this specific {class_name} instance is
2. "shape" – typical shape or geometric form of this {class_name} instance
3. "texture" – common surface textures of this {class_name} instance
4. "color" – typical colors of this {class_name} instance
5. "material" – common construction or composition materials of this {class_name} instance
6. "physical_properties" - choose the most appropriate descriptions from the following list: [heavy/light, rigid/flexible, durable/fragile, smooth/rough, movable/fixed, stable/unstable, solid/liquid/gas, solid/hollow]
7. "functions" – typical roles or uses, what can this {class_name} instance be used for
8. "affordance" - how can this {class_name} instance be used, which part of the object is grasped or manipulated to fulfill its function
9. "common_elements" – other elements frequently associated with this {class_name} instance, what other objects are typically found with this object
10. "common_environmental_context" – contexts or settings where this {class_name} instance is usually found
11. "additional_details" – optional variations, decorative aspects, or other relevant information about this specific {class_name} instance

Instructions:
- Generate ONLY the missing keys listed above
- Be specific and detailed for each key
- Use the object information and existing description to inform your generation
- Return ONLY valid JSON with the missing keys
- Do not include any other text

Generate the missing keys now:
"""
            
            try:
                # Generate missing keys using Gemini
                response = self.model.generate_content(prompt, generation_config={
                    'max_output_tokens': 1500,
                    'temperature': 0.7
                })
                
                if response and response.text:
                    # Parse the response for missing keys
                    generated_keys = self.parse_json_response(response.text)
                    if generated_keys:
                        # Add only the missing keys to the completed description
                        for key in missing_keys:
                            if key in generated_keys:
                                completed_description[key] = generated_keys[key]
                            else:
                                # Fallback for any keys that weren't generated
                                completed_description[key] = self.generate_fallback_key(key, class_name, confidence, bbox)
                    else:
                        # If parsing failed, use fallback for all missing keys
                        for key in missing_keys:
                            completed_description[key] = self.generate_fallback_key(key, class_name, confidence, bbox)
                else:
                    # If API call failed, use fallback for all missing keys
                    for key in missing_keys:
                        completed_description[key] = self.generate_fallback_key(key, class_name, confidence, bbox)
                        
            except Exception as e:
                logger.warning(f"Failed to generate missing keys with Gemini for {class_name}: {e}")
                # Use fallback for all missing keys
                for key in missing_keys:
                    completed_description[key] = self.generate_fallback_key(key, class_name, confidence, bbox)
        
        return completed_description
    
    def generate_fallback_key(self, key: str, class_name: str, confidence: float, bbox: List[float]) -> str:
        """Generate a fallback value for a specific key"""
        bbox_width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
        bbox_height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
        bbox_area = bbox_width * bbox_height
        
        if key == "shape":
            return f"The {class_name} has a typical shape for its class, with dimensions of approximately {bbox_width:.1f} pixels wide by {bbox_height:.1f} pixels tall."
        elif key == "texture":
            return f"The {class_name} has common surface textures typical for objects of this category."
        elif key == "color":
            return f"The {class_name} appears in typical colors for its type, with a color scheme that is characteristic of objects in this category."
        elif key == "material":
            return f"The {class_name} is made of common construction or composition materials typical for objects of this category."
        elif key == "physical_properties":
            # Generate appropriate physical properties based on object type
            if "vehicle" in class_name.lower() or "car" in class_name.lower() or "motorcycle" in class_name.lower() or "bike" in class_name.lower():
                return f"heavy, rigid, durable, smooth, movable, stable, solid"
            elif "person" in class_name.lower() or "human" in class_name.lower() or "woman" in class_name.lower() or "man" in class_name.lower():
                return f"light, flexible, fragile, smooth, movable, stable, solid"
            elif "animal" in class_name.lower() or "giraffe" in class_name.lower() or "dog" in class_name.lower() or "cat" in class_name.lower():
                return f"light, flexible, fragile, smooth, movable, stable, solid"
            elif "tool" in class_name.lower() or "utensil" in class_name.lower() or "pot" in class_name.lower() or "pan" in class_name.lower() or "sheet" in class_name.lower():
                return f"heavy, rigid, durable, smooth, movable, stable, solid"
            elif "food" in class_name.lower() or "biscuit" in class_name.lower() or "bread" in class_name.lower() or "fruit" in class_name.lower():
                return f"light, flexible, fragile, smooth, movable, stable, solid"
            elif "helmet" in class_name.lower() or "hat" in class_name.lower() or "cap" in class_name.lower():
                return f"light, rigid, durable, smooth, movable, stable, solid"
            elif "field" in class_name.lower() or "ground" in class_name.lower() or "dirt" in class_name.lower():
                return f"heavy, rigid, durable, rough, fixed, stable, solid"
            else:
                return f"medium weight, rigid, durable, smooth, movable, stable, solid"
        elif key == "functions":
            return f"The {class_name} serves typical roles or uses, performing what this object can be used for."
        elif key == "affordance":
            return f"The {class_name} can be used in typical ways, with parts that can be grasped or manipulated to fulfill its function."
        elif key == "common_elements":
            return f"The {class_name} has other elements frequently associated with it, including objects typically found with this object."
        elif key == "common_environmental_context":
            return f"The {class_name} is typically found in contexts or settings common to objects of this type."
        elif key == "additional_details":
            return f"This {class_name} instance was detected with confidence {confidence:.3f} and occupies an area of {bbox_area:.1f} square pixels in the image. Additional variations, decorative aspects, or other relevant information may be present."
        else:
            return f"Information about {key} for this {class_name} instance."
    
    def generate_fallback_description(self, class_name: str, confidence: float, detection_data: Dict = None) -> Dict[str, str]:
        """Generate a fallback description when API fails"""
        # Extract additional info if available
        bbox_info = ""
        if detection_data and 'xyxy' in detection_data:
            bbox = detection_data['xyxy']
            # Parse bbox from string representation if needed
            if isinstance(bbox, str):
                # Remove brackets and split by spaces, then convert to float
                bbox_str = bbox.strip('[]')
                bbox = [float(x) for x in bbox_str.split()]
            elif isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], str):
                # Convert string list to float list
                bbox = [float(x) for x in bbox]
            
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                bbox_info = f" (size: {width:.1f}x{height:.1f})"
        
        return {
            "general_description": f"A {class_name} detected with confidence {confidence:.3f}{bbox_info}",
            "shape": f"The {class_name} has a typical shape for its class",
            "texture": f"The {class_name} has a standard texture for its material",
            "color": f"The {class_name} appears in typical colors for its type",
            "material": f"The {class_name} is made of standard materials for its category",
            "physical_properties": f"The {class_name} has typical physical properties for its class",
            "functions": f"The {class_name} serves its intended function",
            "affordance": f"The {class_name} can be interacted with in typical ways",
            "common_elements": f"The {class_name} has standard features for its type",
            "common_environmental_context": f"The {class_name} is found in typical environments",
            "additional_details": f"This {class_name} instance was detected with confidence {confidence:.3f}{bbox_info}"
        }
    
    def process_single_image(self, image_dir: str, output_dir: str) -> bool:
        """
        Process a single image directory and generate descriptions for all detected objects.
        
        Args:
            image_dir: Path to image directory (e.g., /path/to/image_name/)
            output_dir: Output directory for descriptions
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image_name = os.path.basename(image_dir.rstrip('/'))
            logger.info(f"Processing image: {image_name}")
            
            # Find annotation file
            annotation_path = os.path.join(image_dir, 'annotations', f'{image_name}.json')
            if not os.path.exists(annotation_path):
                logger.error(f"Annotation file not found: {annotation_path}")
                return False
            
            # Load annotation data
            with open(annotation_path, 'r') as f:
                annotation_data = json.load(f)
            
            # Find original image
            image_path = os.path.join(image_dir, f'{image_name}.jpg')
            if not os.path.exists(image_path):
                # Try other extensions
                for ext in ['.png', '.jpeg', '.tiff']:
                    alt_path = os.path.join(image_dir, f'{image_name}{ext}')
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
                else:
                    logger.warning(f"Original image not found for {image_name}")
                    image_path = ""  # Will use empty path
            
            # Extract detections
            detections = annotation_data.get('detections', [])
            if not detections:
                logger.warning(f"No detections found for {image_name}")
                return False
            
            # Generate descriptions for each detection
            image_descriptions = {
                "image_name": image_name,
                "image_path": image_path,
                "total_detections": len(detections),
                "object_descriptions": {}
            }
            
            successful_descriptions = 0
            
            for i, detection in enumerate(detections):
                try:
                    class_name = detection.get('class_name', f'object_{i}')
                    confidence = detection.get('confidence', 0.0)
                    bbox = detection.get('xyxy', [0, 0, 100, 100])
                    
                    # Find cropped object image
                    cropped_image_path = ""
                    object_crops_dir = os.path.join(image_dir, 'object_crops', image_name)
                    if os.path.exists(object_crops_dir):
                        # Look for cropped image with this detection index
                        crop_filename = f"obj_{i:03d}_{class_name}_conf{confidence:.2f}.png"
                        crop_path = os.path.join(object_crops_dir, crop_filename)
                        if os.path.exists(crop_path):
                            cropped_image_path = crop_path
                        else:
                            # Try alternative naming patterns
                            for ext in ['.png', '.jpg', '.jpeg']:
                                alt_crop_path = os.path.join(object_crops_dir, f"obj_{i:03d}{ext}")
                                if os.path.exists(alt_crop_path):
                                    cropped_image_path = alt_crop_path
                                    break
                    
                    # Get image context from scene info if available
                    image_context = ""
                    if 'scene_3d_info' in annotation_data:
                        scene_info = annotation_data['scene_3d_info']
                        if isinstance(scene_info, dict):
                            image_context = f"Scene contains {len(detections)} objects with 3D reconstruction"
                    
                    logger.info(f"  Generating description for {class_name} (confidence: {confidence:.3f})")
                    if cropped_image_path:
                        logger.info(f"    Found cropped image: {os.path.basename(cropped_image_path)}")
                    else:
                        logger.warning(f"    No cropped image found for detection {i}")
                    
                    # Parse bbox from string representation if needed
                    parsed_bbox = bbox
                    if isinstance(bbox, str):
                        # Remove brackets and split by spaces, then convert to float
                        bbox_str = bbox.strip('[]')
                        parsed_bbox = [float(x) for x in bbox_str.split()]
                    elif isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], str):
                        # Convert string list to float list
                        parsed_bbox = [float(x) for x in bbox]
                    
                    # Generate description with cropped image path
                    description = self.generate_image_object_description(
                        class_name=class_name,
                        image_path=image_path,
                        bbox=parsed_bbox,
                        confidence=confidence,
                        cropped_image_path=cropped_image_path,
                        image_context=image_context
                    )
                    
                    # Store with detection ID and metadata
                    detection_id = f"obj_{i:03d}"
                    image_descriptions["object_descriptions"][detection_id] = {
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": parsed_bbox,
                        "full_image_path": image_path,
                        "cropped_image_path": cropped_image_path,
                        "description": description,
                        "detection_metadata": {
                            "bbox_width": parsed_bbox[2] - parsed_bbox[0] if len(parsed_bbox) >= 4 else 0,
                            "bbox_height": parsed_bbox[3] - parsed_bbox[1] if len(parsed_bbox) >= 4 else 0,
                            "bbox_area": (parsed_bbox[2] - parsed_bbox[0]) * (parsed_bbox[3] - parsed_bbox[1]) if len(parsed_bbox) >= 4 else 0,
                            "has_cropped_image": bool(cropped_image_path),
                            "detection_index": i
                        }
                    }
                    
                    successful_descriptions += 1
                    
                    # Sleep between API calls
                    time.sleep(self.sleep_sec)
                    
                except Exception as e:
                    logger.error(f"Error processing detection {i} in {image_name}: {e}")
                    continue
            
            # Save image descriptions
            output_path = os.path.join(output_dir, f'{image_name}_object_descriptions.json')
            with open(output_path, 'w') as f:
                json.dump(image_descriptions, f, indent=2)
            
            logger.info(f"Saved {successful_descriptions}/{len(detections)} descriptions for {image_name}")
            return successful_descriptions > 0
            
        except Exception as e:
            logger.error(f"Error processing image {image_dir}: {e}")
            return False
    
    def process_annotation_directory(self, input_dir: str, output_dir: str, 
                                   start_idx: int = 0, end_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all images in the annotation directory.
        
        Args:
            input_dir: Directory containing image subdirectories
            output_dir: Output directory for descriptions
            start_idx: Starting index for processing
            end_idx: Ending index for processing (None for all)
            
        Returns:
            Summary statistics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of image directories
        image_dirs = []
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                # Check if it has annotations
                annotation_dir = os.path.join(item_path, 'annotations')
                if os.path.exists(annotation_dir):
                    image_dirs.append(item_path)
        
        # Sort and slice
        image_dirs.sort()
        if end_idx is not None:
            image_dirs = image_dirs[start_idx:end_idx]
        else:
            image_dirs = image_dirs[start_idx:]
        
        logger.info(f"Found {len(image_dirs)} image directories to process")
        logger.info(f"Processing range: {start_idx} to {end_idx if end_idx else 'end'}")
        
        # Process each image
        successful = 0
        failed = 0
        total_descriptions = 0
        
        for i, image_dir in enumerate(image_dirs):
            logger.info(f"[{i+1}/{len(image_dirs)}] Processing: {os.path.basename(image_dir)}")
            
            if self.process_single_image(image_dir, output_dir):
                successful += 1
                # Count descriptions in this image
                image_name = os.path.basename(image_dir)
                desc_path = os.path.join(output_dir, f'{image_name}_object_descriptions.json')
                if os.path.exists(desc_path):
                    with open(desc_path, 'r') as f:
                        desc_data = json.load(f)
                        total_descriptions += desc_data.get('total_detections', 0)
            else:
                failed += 1
        
        # Generate summary
        summary = {
            "total_images": len(image_dirs),
            "successful": successful,
            "failed": failed,
            "total_descriptions": total_descriptions,
            "output_directory": output_dir
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing complete!")
        logger.info(f"Total images: {len(image_dirs)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total descriptions generated: {total_descriptions}")
        logger.info(f"Output directory: {output_dir}")
        
        return summary

def main():
    """Main function to run image object description generation."""
    parser = argparse.ArgumentParser(description='Generate image-specific object descriptions')
    parser.add_argument('--input_dir', 
                       default="/path/to/project/openimages_unified_output",
                       help='Input directory containing image subdirectories with annotations')
    parser.add_argument('--output_dir', 
                       default="/path/to/project/image_object_descriptions_output",
                       help='Output directory for image object descriptions')
    parser.add_argument('--api_key', 
                       default=os.environ.get("GEMINI_API_KEY"),
                       help='Google Gemini API key')
    parser.add_argument('--model_name', 
                       default="gemini-2.0-flash-exp",
                       help='Gemini model name')
    parser.add_argument('--sleep_sec', 
                       type=float, default=2.0,
                       help='Sleep time between API calls')
    parser.add_argument('--start_idx', 
                       type=int, default=0,
                       help='Starting index for processing')
    parser.add_argument('--end_idx', 
                       type=int, default=None,
                       help='Ending index for processing (None for all)')
    
    args = parser.parse_args()
    
    print("Image-Specific Object Description Generator")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Sleep time: {args.sleep_sec}s")
    print(f"Processing range: {args.start_idx} to {args.end_idx if args.end_idx else 'end'}")
    print("=" * 60)
    
    # Initialize generator
    generator = ImageObjectDescriptionGenerator(
        api_key=args.api_key,
        model_name=args.model_name,
        sleep_sec=args.sleep_sec
    )
    
    # Process all images
    summary = generator.process_annotation_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    
    print(f"\nProcessing completed successfully!")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    main()
