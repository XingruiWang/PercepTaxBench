#!/usr/bin/env python3
"""
Object Description Generator for OpenImages Detected Classes

This script generates detailed descriptions for all object classes detected in the OpenImages dataset
using Google Gemini API. The descriptions follow a structured format beneficial for clustering and 
machine learning applications.

Based on the generate_descriptions.ipynb notebook but adapted for Gemini API.
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import StopCandidateException
from datetime import datetime
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectDescriptionGenerator:
    """Generates detailed descriptions for object classes using Gemini API"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """Initialize the description generator with Gemini API"""
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # Retry configuration
        self.max_retries = 5
        self.base_delay = 1.0
        
        # Initialize dataset statistics
        self.dataset_stats = {
            'total_images_processed': 0,
            'total_detections': 0,
            'unique_classes': 0
        }
        
    def generate_prompt(self, category_name: str) -> str:
        """Generate a structured prompt for object description"""
        return f"""

You are tasked with transforming the name of an object class into a structured description in JSON format. 
The goal is to represent each object class consistently with the same set of attributes, so that the output is machine-readable and uniform for clustering and machine learning tasks.

### Task
- Input: the name of a class of object (e.g., "wall", "building", "tree").
- Output: a JSON object where the class name is the key, and its value is another JSON object containing exactly **11 attributes**.
- Every output must contain **all 11 keys** listed below, even if some values are empty.  
- Values should be concise natural language descriptions, not lists of bullet points.
- please keep the keys in the json file as well.
- The JSON must be syntactically valid.

### Required 11 keys
1. "general_description" – short summary of what the class of object is
2. "shape" – typical shape or geometric form
3. "texture" – common surface textures, what it feels like when touched
4. "color" – typical colors of the object that are distinctive to the object
5. "material" – common construction or composition materials specific to the object components
6. "physical_properties" - choose the most appropriate descriptions from the following list: [heavy/light, rigid/flexible, durable/fragile, smooth/rough, movable/fixed, stable/unstable, solid/liquid/gas, solid/hollow]
   **CRITICAL: For physical_properties, you MUST choose only ONE option from each pair separated by "/". 
   CORRECT examples: "heavy, rigid, durable, rough, fixed" 
   INCORRECT examples: "heavy, light, durable, fragile" (contradictory pairs)**
7. "functions" – typical roles or uses, what can the object be used for
8. "affordance" - how can the object be used, which part of the object is manipulated to fulfill its function
9. "common_elements" – other elements frequently associated with the object, what other objects are typically found with the object
10. "common_environmental_context" – contexts or settings where the object is usually found
11. "additional_details" – optional variations, decorative aspects, or other relevant information

Please apply this detailed, structured format to describe "{category_name}".
"""

    def string_to_list(self, description: str) -> List[str]:
        """Convert description string to list of descriptors"""
        descriptors = []
        
        # First, try to find JSON content in the response
        import json
        import re
        
        # Look for JSON content between curly braces
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, description)
        
        for json_str in json_matches:
            try:
                parsed_data = json.loads(json_str)
                
                # Extract values from the JSON object
                if isinstance(parsed_data, dict):
                    for key, value in parsed_data.items():
                        if isinstance(value, dict):
                            # Extract all string values from the nested object
                            for attr_key, attr_value in value.items():
                                if isinstance(attr_value, str) and attr_value.strip():
                                    descriptors.append(attr_value.strip())
                        elif isinstance(value, str) and value.strip():
                            descriptors.append(value.strip())
                            
            except json.JSONDecodeError:
                continue
        
        # If no JSON found, try to extract bullet points as fallback
        if not descriptors:
            for descriptor in description.split('\n'):
                if descriptor.strip() and descriptor.startswith('- '):
                    clean_desc = descriptor[2:].strip()
                    if len(clean_desc) >= 5:
                        descriptors.append(clean_desc)
        
        # If still no descriptors, try to extract any meaningful text
        if not descriptors:
            lines = description.split('\n')
            for line in lines:
                line = line.strip()
                # Look for lines that contain descriptive content
                if (len(line) >= 10 and 
                    not line.startswith('#') and 
                    not line.startswith('```') and
                    not line.startswith('{') and
                    not line.startswith('}')):
                    descriptors.append(line)
        
        return descriptors
    
    def parse_json_response(self, description: str) -> Dict[str, str]:
        """Parse JSON response and return structured data with keys"""
        import json
        import re
        
        # Look for JSON content between curly braces
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, description)
        
        for json_str in json_matches:
            try:
                parsed_data = json.loads(json_str)
                
                # Extract the nested object with the category name as key
                if isinstance(parsed_data, dict):
                    for category_name, category_data in parsed_data.items():
                        if isinstance(category_data, dict):
                            # Return the structured data with proper keys
                            return category_data
                        elif isinstance(category_data, str):
                            # If the value is a string, create a simple structure
                            return {"description": category_data}
                            
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try to extract from markdown code blocks
        if "```json" in description:
            try:
                # Extract content between ```json and ```
                json_start = description.find("```json") + 7
                json_end = description.find("```", json_start)
                if json_end > json_start:
                    json_content = description[json_start:json_end].strip()
                    parsed_data = json.loads(json_content)
                    
                    # Extract the nested object with the category name as key
                    if isinstance(parsed_data, dict):
                        for category_name, category_data in parsed_data.items():
                            if isinstance(category_data, dict):
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
                                 "physical properties", "functions", "affordance", "common_elements", 
                                 "common_environmental_context", "additional_details"]:
                        continue
                    result[key] = value.strip()
                if result:
                    return result
        except Exception:
            pass
        
        # If no valid JSON found, return empty dict
        return {}

    def retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                
                # Check for quota limit errors
                if "429" in error_msg and "quota" in error_msg.lower():
                    if attempt == self.max_retries - 1:
                        logger.error(f"Quota limit reached for {func.__name__}. Skipping this request.")
                        raise e
                    
                    # Longer delay for quota limits
                    delay = self.base_delay * (2 ** attempt) * 3 + random.uniform(5, 10)
                    logger.warning(f"Quota limit hit (attempt {attempt + 1}). Waiting {delay:.2f}s before retry...")
                    time.sleep(delay)
                else:
                    if attempt == self.max_retries - 1:
                        raise e
                    
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)

    def generate_description(self, category_name: str) -> tuple[List[str], Dict[str, str]]:
        """Generate description for a single category"""
        try:
            prompt = self.generate_prompt(category_name)
            
            response = self.retry_with_backoff(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent output
                    max_output_tokens=2000  # Increased for structured JSON responses
                )
            )
            
            try:
                if response.text:
                    logger.info(f"Raw response for '{category_name}': {response.text[:200]}...")
                    # Parse structured JSON response
                    structured_data = self.parse_json_response(response.text)
                    if structured_data:
                        # Convert structured data to list format for compatibility
                        descriptors = list(structured_data.values())
                        logger.info(f"Generated {len(descriptors)} descriptors for '{category_name}'")
                        return descriptors, structured_data
                    else:
                        # Fallback to old method if JSON parsing fails
                        descriptors = self.string_to_list(response.text)
                        logger.info(f"Generated {len(descriptors)} descriptors for '{category_name}' (fallback)")
                        return descriptors, {}
                else:
                    # Check if response was blocked
                    finish_reason = getattr(response, 'finish_reason', 'unknown')
                    if finish_reason == 2:
                        logger.warning(f"Response blocked by safety filter for '{category_name}' (finish_reason: {finish_reason})")
                    else:
                        logger.error(f"No response text for '{category_name}' - finish_reason: {finish_reason}")
                    return [], {}
            except Exception as text_error:
                error_msg = str(text_error)
                if "finish_reason" in error_msg and "2" in error_msg:
                    logger.warning(f"Response blocked by safety filter for '{category_name}' - skipping")
                else:
                    logger.error(f"Error accessing response text for '{category_name}': {text_error}")
                return [], {}
                
        except Exception as e:
            error_msg = str(e)
            if "finish_reason" in error_msg and "2" in error_msg:
                logger.warning(f"Response blocked by safety filter for '{category_name}' - skipping")
            else:
                logger.error(f"Failed to generate description for '{category_name}': {e}")
            return [], {}

    def load_categories_from_file(self, file_path: str) -> List[str]:
        """Load category names from a text file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Clean up lines: remove whitespace, empty lines, and duplicates
            categories = []
            for line in lines:
                category = line.strip()
                if category and category not in categories:
                    categories.append(category)
            
            logger.info(f"Loaded {len(categories)} unique categories from {file_path}")
            return categories
            
        except Exception as e:
            logger.error(f"Failed to load categories from {file_path}: {e}")
            return []

    def extract_detected_classes(self, output_dir: str) -> List[str]:
        """Extract all unique class names from OpenImages output"""
        detected_classes = set()
        total_images = 0
        total_detections = 0
        
        # Find all JSON files in the output directory
        json_files = list(Path(output_dir).rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Skip dataset statistics file
                if json_file.name == 'dataset_statistics.json':
                    continue
                    
                total_images += 1
                
                # Extract class names from detections
                if 'detections' in data:
                    for detection in data['detections']:
                        if 'class_name' in detection:
                            detected_classes.add(detection['class_name'])
                            total_detections += 1
                            
            except Exception as e:
                logger.warning(f"Failed to process {json_file}: {e}")
                continue
        
        # Convert to sorted list
        class_list = sorted(list(detected_classes))
        logger.info(f"Extracted {len(class_list)} unique classes from {total_images} images with {total_detections} total detections")
        
        # Store statistics for summary
        self.dataset_stats = {
            'total_images_processed': total_images,
            'total_detections': total_detections,
            'unique_classes': len(class_list)
        }
        
        return class_list

    def generate_descriptions_for_classes(self, class_list: List[str], 
                                        output_file: str = "object_descriptions.json",
                                        sleep_sec: float = 1.0) -> Dict[str, Dict[str, str]]:
        """Generate descriptions for classes and save to file, skipping existing ones"""
        # Load existing descriptions
        existing_descriptions = self.load_existing_descriptions(output_file)
        logger.info(f"Loaded {len(existing_descriptions)} existing descriptions from {output_file}")
        
        # Find classes that still need descriptions
        remaining_classes = [cls for cls in class_list if cls not in existing_descriptions]
        
        if not remaining_classes:
            logger.info("All classes already have descriptions!")
            return existing_descriptions
        
        logger.info(f"Found {len(remaining_classes)} classes that need descriptions")
        new_descriptors = {}
        
        for i, category in enumerate(remaining_classes, 1):
            logger.info(f"Processing {i}/{len(remaining_classes)}: {category}")
            
            category_descriptors, structured_data = self.generate_description(category)
            if category_descriptors:
                if structured_data:
                    new_descriptors[category] = structured_data
                else:
                    # Fallback to list format if structured parsing fails
                    new_descriptors[category] = {"descriptions": category_descriptors}
                
                # Save progress incrementally after each successful generation
                current_all_descriptors = {**existing_descriptions, **new_descriptors}
                self.save_progress(current_all_descriptors, output_file, category, len(class_list))
            
            # Sleep between requests to avoid rate limiting
            if i < len(remaining_classes):
                time.sleep(sleep_sec)
        
        # Merge with existing descriptions
        all_descriptors = {**existing_descriptions, **new_descriptors}
        
        # Save complete file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(all_descriptors, f, indent=4)
        
        logger.info(f"Added {len(new_descriptors)} new descriptions")
        logger.info(f"Total descriptions saved: {len(all_descriptors)} classes to {output_path}")
        return all_descriptors
    
    def save_progress(self, descriptions: Dict[str, Dict[str, str]], output_file: str, current_class: str, total_classes: int):
        """Save progress incrementally after each successful generation"""
        try:
            output_path = Path(output_file)
            
            # Load existing data if file exists
            existing_data = {}
            if output_path.exists():
                try:
                    with open(output_path, 'r') as f:
                        existing_data = json.load(f)
                except:
                    pass
            
            # Update with new descriptions
            existing_data.update(descriptions)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
            
            logger.info(f"Progress saved: {len(existing_data)}/{total_classes} classes completed (current: {current_class})")
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def load_existing_descriptions(self, file_path: str) -> Dict[str, Dict[str, str]]:
        """Load existing descriptions from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if data is in new structured format (dict with 11 keys) or old format (list)
            cleaned_data = {}
            incomplete_count = 0
            
            for category, content in data.items():
                if isinstance(content, dict):
                    # New structured format - check if all 11 required keys are present
                    required_keys = [
                        "general_description", "shape", "texture", "color", "material",
                        "physical_properties", "functions", "affordance", "common_elements",
                        "common_environmental_context", "additional_details"
                    ]
                    
                    # Handle both underscore and space versions of physical_properties
                    if "physical_properties" not in content and "physical properties" in content:
                        content["physical_properties"] = content["physical properties"]
                    
                    if all(key in content for key in required_keys):
                        # Check if any descriptions are incomplete
                        has_incomplete = False
                        for key, desc in content.items():
                            clean_desc = desc.strip()
                            if len(clean_desc) < 10:  # More lenient - only reject very short descriptions
                                incomplete_count += 1
                                has_incomplete = True
                        
                        if not has_incomplete:
                            cleaned_data[category] = content
                        else:
                            logger.warning(f"Skipping {category} due to incomplete descriptions")
                    else:
                        logger.warning(f"Skipping {category} due to missing required keys")
                        incomplete_count += 1
                        
                elif isinstance(content, list):
                    # Old format - convert to new format if possible
                    cleaned_descriptions = []
                    for desc in content:
                        clean_desc = desc.strip()
                        if len(clean_desc) >= 20 and clean_desc.endswith(('.', '!', '?')):
                            cleaned_descriptions.append(clean_desc)
                        elif len(clean_desc) >= 30:
                            cleaned_descriptions.append(clean_desc)
                        else:
                            incomplete_count += 1
                    
                    if cleaned_descriptions:
                        # Convert to new format
                        cleaned_data[category] = {
                            "general_description": cleaned_descriptions[0] if cleaned_descriptions else "",
                            "shape": cleaned_descriptions[1] if len(cleaned_descriptions) > 1 else "",
                            "texture": cleaned_descriptions[2] if len(cleaned_descriptions) > 2 else "",
                            "color": cleaned_descriptions[3] if len(cleaned_descriptions) > 3 else "",
                            "material": cleaned_descriptions[4] if len(cleaned_descriptions) > 4 else "",
                            "physical_properties": cleaned_descriptions[5] if len(cleaned_descriptions) > 5 else "",
                            "functions": cleaned_descriptions[6] if len(cleaned_descriptions) > 6 else "",
                            "affordance": cleaned_descriptions[7] if len(cleaned_descriptions) > 7 else "",
                            "common_elements": cleaned_descriptions[8] if len(cleaned_descriptions) > 8 else "",
                            "common_environmental_context": cleaned_descriptions[9] if len(cleaned_descriptions) > 9 else "",
                            "additional_details": cleaned_descriptions[10] if len(cleaned_descriptions) > 10 else ""
                        }
            
            if incomplete_count > 0:
                logger.info(f"Cleaned up {incomplete_count} incomplete descriptions from existing file")
            
            logger.info(f"Loaded {len(cleaned_data)} existing descriptions from {file_path}")
            return cleaned_data
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Failed to load existing descriptions: {e}")
            return {}



def main():
    parser = argparse.ArgumentParser(description='Generate object descriptions using Gemini API')
    parser.add_argument('--api_key', type=str, required=True, 
                       help='Google Gemini API key')
    parser.add_argument('--output_dir', type=str, 
                       default='/path/to/project/openimages_unified_output',
                       help='Path to OpenImages output directory')
    parser.add_argument('--input_file', type=str, 
                       help='Path to text file containing category names (alternative to output_dir)')
    parser.add_argument('--output_file', type=str, default='object_descriptions.json',
                       help='Output JSON file for descriptions')
    parser.add_argument('--sleep_sec', type=float, default=2.0,
                       help='Sleep time between API requests')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash',
                       help='Gemini model to use')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ObjectDescriptionGenerator(args.api_key, args.model)
    
    # Get class list from either input file or OpenImages output
    if args.input_file:
        class_list = generator.load_categories_from_file(args.input_file)
    else:
        class_list = generator.extract_detected_classes(args.output_dir)
    
    if not class_list:
        logger.error("No classes found in output directory")
        return
    
    logger.info(f"Found {len(class_list)} unique classes: {class_list[:10]}...")
    
    # Generate descriptions (automatically resumes from existing file)
    descriptions = generator.generate_descriptions_for_classes(class_list, args.output_file, args.sleep_sec)
    
    # Count incomplete descriptions in final output
    incomplete_descriptions = []
    total_descriptions = 0
    for category, desc_list in descriptions.items():
        for desc in desc_list:
            total_descriptions += 1
            clean_desc = desc.strip()
            if len(clean_desc) < 20 or (len(clean_desc) < 30 and not clean_desc.endswith(('.', '!', '?'))):
                incomplete_descriptions.append((category, clean_desc))
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("OBJECT DESCRIPTION GENERATION SUMMARY")
    print("="*60)
    print(f" Dataset Statistics:")
    print(f"   • Total images processed: {generator.dataset_stats['total_images_processed']:,}")
    print(f"   • Total object detections: {generator.dataset_stats['total_detections']:,}")
    print(f"   • Unique object categories: {generator.dataset_stats['unique_classes']:,}")
    print(f"   • Average detections per image: {generator.dataset_stats['total_detections'] / max(generator.dataset_stats['total_images_processed'], 1):.1f}")
    print()
    print(f" Description Generation:")
    print(f"   • Categories with descriptions: {len(descriptions):,}")
    print(f"   • Total individual descriptions: {total_descriptions:,}")
    print(f"   • Incomplete descriptions found: {len(incomplete_descriptions):,}")
    print(f"   • Coverage percentage: {(len(descriptions) / max(generator.dataset_stats['unique_classes'], 1)) * 100:.1f}%")
    print(f"   • Output file: {args.output_file}")
    print()
    if incomplete_descriptions:
        print(f" Incomplete Descriptions (first 5):")
        for i, (cat, desc) in enumerate(incomplete_descriptions[:5]):
            print(f"   • {cat}: \"{desc}\"")
        if len(incomplete_descriptions) > 5:
            print(f"   • ... and {len(incomplete_descriptions) - 5} more")
        print()
    print(f" OpenImages Unified Output Directory:")
    print(f"   • Path: {args.output_dir}")
    print(f"   • Contains: {generator.dataset_stats['total_images_processed']} processed images")
    print("="*60)
    
    # Also log the summary
    logger.info(f"Generation complete! Generated descriptions for {len(descriptions)} classes")
    logger.info(f"Dataset processed: {generator.dataset_stats['total_images_processed']} images, {generator.dataset_stats['total_detections']} detections, {generator.dataset_stats['unique_classes']} unique classes")
    logger.info(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    main()
