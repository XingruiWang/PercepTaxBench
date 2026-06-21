#!/usr/bin/env python3
"""
Gemini-based Object Detection with Taxonomy Integration
Uses Gemini API to detect objects from SM_names in images and return taxonomy attributes with bounding boxes.
"""

import json
import os
import base64
import logging
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import io
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeminiObjectDetector:
    def __init__(self, api_key: str = None):
        """Initialize the Gemini Object Detector."""
        self.api_keys = [
            ""
        ]
        self.current_key_index = 0
        self.api_key = api_key or self.api_keys[0]
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None
    
    def _rotate_api_key(self):
        """Rotate to the next API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None
    
    def _load_sm_names(self, sm_names_file: str) -> List[str]:
        """Load SM object names from file."""
        try:
            with open(sm_names_file, 'r') as f:
                sm_objects = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(sm_objects)} SM objects")
            return sm_objects
        except FileNotFoundError:
            logger.error(f"SM names file not found: {sm_names_file}")
            return []
    
    def _load_taxonomy(self, taxonomy_file: str) -> Dict[str, Any]:
        """Load clustering taxonomy from final_taxonomy.json."""
        try:
            with open(taxonomy_file, 'r') as f:
                taxonomy = json.load(f)
            logger.info(f"Loaded taxonomy with {len(taxonomy)} keys")
            return taxonomy
        except FileNotFoundError:
            logger.error(f"Taxonomy file not found: {taxonomy_file}")
            return {}
    
    def _create_detection_prompt(self, sm_objects: List[str], taxonomy: Dict[str, Any]) -> str:
        """Create the prompt for Gemini object detection."""
        
        # Create SM objects list for the prompt
        sm_objects_text = "\n".join([f"- {obj}" for obj in sm_objects])
        
        # Create taxonomy summary for the prompt
        taxonomy_summary = {}
        for key, clusters in taxonomy.items():
            taxonomy_summary[key] = {}
            for cluster_name, cluster_data in clusters.items():
                objects_in_cluster = cluster_data.get("objects", [])
                # Only include clusters that have SM objects
                sm_objects_in_cluster = [obj for obj in objects_in_cluster if obj in sm_objects]
                if sm_objects_in_cluster:
                    taxonomy_summary[key][cluster_name] = {
                        "description": cluster_data.get("description", ""),
                        "objects": sm_objects_in_cluster
                    }
        
        taxonomy_text = json.dumps(taxonomy_summary, indent=2)
        
        prompt = f"""You are an expert object detection system with precise bounding box prediction capabilities. Your task is to detect objects from the provided SM object list in the given image and return detailed information about each detected object.

SM OBJECTS TO DETECT:
{sm_objects_text}

TAXONOMY CATEGORIES AND CLUSTERS:
{taxonomy_text}

CRITICAL BOUNDING BOX INSTRUCTIONS:
1. Analyze the image carefully and identify ALL objects that match the SM object list above
2. For each detected object, provide ACCURATE bounding box coordinates:
   - (x1, y1) = top-left corner of the object (leftmost and topmost pixel)
   - (x2, y2) = bottom-right corner of the object (rightmost and bottommost pixel)
   - The bounding box MUST tightly enclose the ENTIRE visible object
   - Include all parts of the object within the box (legs of chairs, handles of cups, etc.)
   - Do NOT crop out important parts of the object
   - Coordinates must be in pixels, starting from (0,0) at top-left of image

3. For EACH detected object, also provide:
   - The exact object name from the SM list (use lowercase if applicable)
   - All taxonomy categories and clusters where this object appears
   - The cluster descriptions for each category

4. Quality requirements:
   - Only detect objects that are clearly visible and identifiable
   - Be precise and careful with bounding box measurements
   - Double-check that each bounding box fully contains the object
   - Include ALL taxonomy categories where the object appears

OUTPUT FORMAT:
Return a JSON object with this exact structure:
include all taxonomy categories and clusters where the object appears, including  texture, function, common_environment, common_elements, affordance, shape, material, and physical_properties.
{{
    "detected_objects": [
        {{
            "object_name": "exact name from SM list",
            "bounding_box": {{
                "x1": 100,
                "y1": 50,
                "x2": 200,
                "y2": 150
            }},
            "taxonomy_attributes": {{
                "function": [
                    {{
                        "cluster_name_1": "cluster name", 
                        "cluster_description_1": "description",
                        "object_in_cluster": true
                    },
                    
                    },
                    {{
                        "cluster_name_2": "cluster name", 
                        "cluster_description_2": "description",
                        "object_in_cluster": true
                    }
                    ...}
                ],
                "common_environment": [
                    {{
                        "cluster_name": "cluster name",
                        "cluster_description": "description", 
                        "object_in_cluster": true
                    }, 
                    cluster_name_2": "cluster name", 
                        "cluster_description_2": "description",
                        "object_in_cluster": true
                    },
                    ...
                    }
                ],
                ],
                "affordance": [
                    {{
                        "cluster_name": "cluster name",
                        "cluster_description": "description",
                        "object_in_cluster": true
                    }}
                ],
                "shape": [
                    {{
                        "cluster_name": "cluster name",
                        "cluster_description": "description",
                        "object_in_cluster": true
                    }}
                ],
                "material": [
                    {{
                        "cluster_name": "cluster name",
                        "cluster_description": "description",
                        "object_in_cluster": true
                    }}
                ],
                "physical_properties": [
                    {{
                        "cluster_name": "cluster name",
                        "cluster_description": "description",
                        "object_in_cluster": true
                    }}
                ]
            }}
        }}
    ]
}}

IMPORTANT REMINDERS:
- Only include objects that are actually visible in the image
- Use exact object names from the SM list (match case and spelling exactly)
- Bounding boxes must be ACCURATE and fully contain the object
- Check each corner of the bounding box to ensure it captures the full object
- If a chair has legs, make sure y2 extends to the bottom of the legs
- If a bookshelf is tall, make sure y2 extends to the base
- Include ALL taxonomy categories where the object appears
- Be accurate and thorough in your analysis

EXAMPLE:
If you see a chair that spans from x=100 to x=300 horizontally and from y=400 to y=700 vertically (including all legs and back), 
the bounding box should be: {{"x1": 100, "y1": 400, "x2": 300, "y2": 700}}
NOT {{"x1": 150, "y1": 450, "x2": 250, "y2": 600}} (this would cut off parts of the chair)"""

        return prompt
    
    def detect_objects(self, image_path: str, sm_names_file: str, taxonomy_file: str) -> Dict[str, Any]:
        """
        Detect objects in an image using Gemini API.
        
        Args:
            image_path: Path to the input image
            sm_names_file: Path to SM_names.txt file
            taxonomy_file: Path to final_taxonomy.json file
            
        Returns:
            Dictionary containing detection results with taxonomy information
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {}
        
        # Load SM objects and taxonomy
        sm_objects = self._load_sm_names(sm_names_file)
        taxonomy = self._load_taxonomy(taxonomy_file)
        
        if not sm_objects or not taxonomy:
            logger.error("Failed to load SM objects or taxonomy")
            return {}
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except Exception as e:
            logger.error(f"Failed to get image dimensions: {e}")
            return {}
        
        # Create prompt
        prompt = self._create_detection_prompt(sm_objects, taxonomy)
        
        # Call Gemini API
        try:
            # Load image for Gemini
            image = Image.open(image_path)
            
            # Generate content
            response = self.model.generate_content([prompt, image])
            response_text = response.text
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start:json_end].strip()
                results = json.loads(json_text)
            else:
                logger.error("No JSON found in Gemini response")
                return {}
            
            # Add metadata
            results["metadata"] = {
                "image_path": image_path,
                "image_dimensions": {"width": image_width, "height": image_height},
                "total_objects_detected": len(results.get("detected_objects", [])),
                "sm_objects_available": len(sm_objects)
            }
            
            logger.info(f"Detected {len(results.get('detected_objects', []))} objects")
            return results
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            # Try with API key rotation
            try:
                self._rotate_api_key()
                response = self.model.generate_content([prompt, image])
                response_text = response.text
                
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_text = response_text[json_start:json_end].strip()
                    results = json.loads(json_text)
                else:
                    logger.error("No JSON found in Gemini response after retry")
                    return {}
                
                results["metadata"] = {
                    "image_path": image_path,
                    "image_dimensions": {"width": image_width, "height": image_height},
                    "total_objects_detected": len(results.get("detected_objects", [])),
                    "sm_objects_available": len(sm_objects)
                }
                
                logger.info(f"Detected {len(results.get('detected_objects', []))} objects (retry successful)")
                return results
                
            except Exception as retry_error:
                logger.error(f"Gemini API retry also failed: {retry_error}")
                return {}
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save detection results to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_file}")
    
    def draw_bounding_boxes(self, image_path: str, results: Dict[str, Any], output_image_path: str):
        """Draw bounding boxes on the image and save visualization."""
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        
        for idx, obj in enumerate(results.get("detected_objects", [])):
            bbox = obj.get("bounding_box", {})
            x1, y1, x2, y2 = bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")
            
            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue
            
            color = colors[idx % len(colors)]
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            label = obj.get("object_name", "Unknown")
            
            text_bbox = draw.textbbox((x1, y1), label)
            text_height = text_bbox[3] - text_bbox[1]
            text_width = text_bbox[2] - text_bbox[0]
            
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - text_height - 2), label, fill='white')
        
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        image.save(output_image_path)
        logger.info(f"Visualization saved to: {output_image_path}")


def main():
    """Main function for testing the Gemini Object Detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemini Object Detection with Taxonomy Integration")
    parser.add_argument("--image", help="Path to input image (for single image)")
    parser.add_argument("--image_dir", help="Path to directory containing images (for batch processing)")
    parser.add_argument("--output_dir", default="clustering/results/gemini_detection_results", help="Output directory for results")
    parser.add_argument("--sm_names", default="SM_names.txt", help="Path to SM_names.txt")
    parser.add_argument("--taxonomy", default="clustering/results/final_taxonomy.json", help="Path to final_taxonomy.json")
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be specified")
    
    # Initialize detector
    detector = GeminiObjectDetector()
    
    # Determine image list
    if args.image:
        image_files = [args.image]
    else:
        image_files = glob.glob(os.path.join(args.image_dir, "*.png")) + \
                     glob.glob(os.path.join(args.image_dir, "*.jpg")) + \
                     glob.glob(os.path.join(args.image_dir, "*.jpeg"))
        image_files.sort()
        logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in image_files:
        logger.info(f"\nProcessing: {image_path}")
        
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        
        image_output_dir = os.path.join(args.output_dir, image_basename)
        os.makedirs(image_output_dir, exist_ok=True)
        
        json_output = os.path.join(image_output_dir, f"{image_basename}_detection.json")
        viz_output = os.path.join(image_output_dir, f"{image_basename}_bbox.png")
        
        results = detector.detect_objects(image_path, args.sm_names, args.taxonomy)
        
        if results:
            detector.save_results(results, json_output)
            detector.draw_bounding_boxes(image_path, results, viz_output)
            
            print(f"\nDetection Summary for {image_basename}:")
            print(f"Total objects detected: {results['metadata']['total_objects_detected']}")
            print(f"Image dimensions: {results['metadata']['image_dimensions']}")
            
            for obj in results.get("detected_objects", []):
                print(f"\nObject: {obj['object_name']}")
                print(f"Bounding box: ({obj['bounding_box']['x1']}, {obj['bounding_box']['y1']}) to ({obj['bounding_box']['x2']}, {obj['bounding_box']['y2']})")
                
                for category, clusters in obj.get("taxonomy_attributes", {}).items():
                    if clusters:
                        print(f"  {category}: {len(clusters)} clusters")
        else:
            logger.warning(f"No objects detected or detection failed for {image_path}")


if __name__ == "__main__":
    main()
