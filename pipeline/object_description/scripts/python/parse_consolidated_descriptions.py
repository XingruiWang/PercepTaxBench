#!/usr/bin/env python3
"""
Parse consolidated object descriptions using the same logic as the original parsing script.
"""

import json
import argparse
import google.generativeai as genai
from pathlib import Path
import logging
from typing import Dict, List
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_consolidated_descriptions(descriptions_file: str) -> Dict[str, Dict]:
    """Load consolidated object descriptions."""
    with open(descriptions_file, 'r') as f:
        descriptions = json.load(f)
    return descriptions

def load_existing_parsed(parsed_file: str) -> Dict[str, Dict]:
    """Load existing parsed descriptions."""
    try:
        with open(parsed_file, 'r') as f:
            parsed = json.load(f)
        return parsed
    except FileNotFoundError:
        logger.warning(f"Existing parsed file not found: {parsed_file}")
        return {}

def parse_object_description(object_name: str, description: Dict[str, str], api_key: str) -> Dict[str, str]:
    """Parse a single object description using Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create the description text
        desc_text = f"""
Object: {object_name}

General Description: {description.get('general_description', '')}
Shape: {description.get('shape', '')}
Texture: {description.get('texture', '')}
Color: {description.get('color', '')}
Material: {description.get('material', '')}
Physical Properties: {description.get('physical_properties', '')}
Functions: {description.get('functions', '')}
Affordance: {description.get('affordance', '')}
Common Elements: {description.get('common_elements', '')}
Common Environmental Context: {description.get('common_environmental_context', '')}
Additional Details: {description.get('additional_details', '')}
"""
        
        prompt = f"""Parse the following object description and extract specific attributes into separate fields. The description contains mixed concepts that need to be separated into distinct attributes.

{desc_text}

Please parse this description and return a JSON object with the following structure:
{{
    "shape": "Extract only shape-related information",
    "texture": "Extract only texture-related information", 
    "color": "Extract only color-related information",
    "material": "Extract only material-related information",
    "physical_properties": "Extract only physical properties (e.g., heavy, light, rigid, flexible, etc.)",
    "functions": "Extract only function-related information",
    "affordance": "Extract only affordance-related information",
    "common_elements": "Extract only common elements/components",
    "common_environmental_context": "Extract only environmental context information",
    "additional_details": "Extract any other relevant details"
}}

Focus on separating the mixed concepts into distinct, focused attributes. Each field should contain only information relevant to that specific attribute."""

        response = model.generate_content(prompt)
        
        # Check if response was blocked
        if response.candidates and response.candidates[0].finish_reason == 8:
            logger.warning(f"Content blocked for {object_name}, using original description")
            return description
        
        # Try to extract JSON from response
        response_text = response.text.strip()
        
        # Look for JSON in the response
        if response_text.startswith('{') and response_text.endswith('}'):
            return json.loads(response_text)
        else:
            # Try to find JSON within the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.error(f"Could not parse JSON for {object_name}: {response_text[:200]}...")
                return description
                
    except Exception as e:
        logger.error(f"Error parsing description for {object_name}: {e}")
        return description

def parse_consolidated_descriptions(descriptions: Dict[str, Dict],
                                  existing_parsed: Dict[str, Dict],
                                  api_key: str,
                                  output_file: str):
    """Parse consolidated object descriptions."""
    
    # Create output dictionary starting with existing parsed descriptions
    parsed_descriptions = existing_parsed.copy()
    
    # Identify objects that need parsing
    objects_to_parse = []
    for obj_name, description in descriptions.items():
        if obj_name not in parsed_descriptions:
            objects_to_parse.append((obj_name, description))
    
    logger.info(f"Need to parse {len(objects_to_parse)} descriptions")
    
    # Parse descriptions
    for i, (obj_name, description) in enumerate(objects_to_parse, 1):
        logger.info(f"Parsing description {i}/{len(objects_to_parse)}: {obj_name}")
        
        parsed_description = parse_object_description(obj_name, description, api_key)
        parsed_descriptions[obj_name] = parsed_description
        
        # Save progress every 10 objects
        if i % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(parsed_descriptions, f, indent=2)
            logger.info(f"Saved progress: {i}/{len(objects_to_parse)} descriptions")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(parsed_descriptions, f, indent=2)
    
    logger.info(f"Parsed {len(objects_to_parse)} descriptions")
    logger.info(f"Total parsed descriptions: {len(parsed_descriptions)}")
    logger.info(f"Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Parse consolidated object descriptions")
    parser.add_argument("--descriptions_file", required=True, help="Path to consolidated descriptions JSON")
    parser.add_argument("--output_file", required=True, help="Output file for parsed descriptions")
    parser.add_argument("--api_key", required=True, help="Gemini API key")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading consolidated descriptions...")
    descriptions = load_consolidated_descriptions(args.descriptions_file)
    
    logger.info("Loading existing parsed descriptions...")
    existing_parsed = load_existing_parsed(args.output_file)
    
    # Parse descriptions
    parse_consolidated_descriptions(
        descriptions,
        existing_parsed,
        args.api_key,
        args.output_file
    )

if __name__ == "__main__":
    main()
