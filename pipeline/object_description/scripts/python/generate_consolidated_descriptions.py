#!/usr/bin/env python3
"""
Generate descriptions for consolidated objects, reusing existing descriptions when available.
"""

import json
import argparse
import google.generativeai as genai
from pathlib import Path
import logging
from typing import Dict, List, Set

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_consolidated_objects(consolidated_file: str) -> List[str]:
    """Load the list of consolidated objects."""
    with open(consolidated_file, 'r') as f:
        objects = [line.strip() for line in f.readlines() if line.strip()]
    return objects

def load_consolidation_mapping(mapping_file: str) -> Dict[str, str]:
    """Load the consolidation mapping (original -> consolidated)."""
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    return mapping

def load_existing_descriptions(descriptions_file: str) -> Dict[str, Dict]:
    """Load existing object descriptions."""
    try:
        with open(descriptions_file, 'r') as f:
            descriptions = json.load(f)
        return descriptions
    except FileNotFoundError:
        logger.warning(f"Existing descriptions file not found: {descriptions_file}")
        return {}


def generate_object_description(object_name: str, api_key: str) -> Dict[str, str]:
    """Generate description for a single object using Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""Generate a comprehensive description for the object "{object_name}" with the following attributes:

1. "general_description" - provide a detailed description of what this object is, its primary function, and common characteristics
2. "shape" - describe the typical shape, form, and geometric properties
3. "texture" - describe the surface texture, material feel, and tactile properties
4. "color" - describe typical colors, color patterns, and visual appearance
5. "material" - describe what materials this object is commonly made from
6. "physical_properties" - choose the most appropriate descriptions from the following list: [heavy/light, rigid/flexible, durable/fragile, smooth/rough, movable/fixed, stable/unstable, solid/liquid/gas, solid/hollow]
   **CRITICAL: For physical_properties, you MUST choose only ONE option from each pair separated by "/". 
   CORRECT examples: "heavy, rigid, durable, rough, fixed" 
   INCORRECT examples: "heavy, light, durable, fragile" (contradictory pairs)**
7. "functions" - describe the primary and secondary functions, uses, and purposes
8. "affordance" - describe what actions or interactions this object affords to users
9. "common_elements" - describe common components, parts, or elements typically found in this object
10. "common_environmental_context" - describe where this object is commonly found or used
11. "additional_details" - any other relevant characteristics, variations, or notable features

Return the response as a JSON object with these exact keys and string values."""

        response = model.generate_content(prompt)
        
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
                return None
                
    except Exception as e:
        logger.error(f"Error generating description for {object_name}: {e}")
        return None


def generate_consolidated_descriptions(consolidated_objects: List[str],
                                     consolidation_mapping: Dict[str, str],
                                     existing_descriptions: Dict[str, Dict],
                                     api_key: str,
                                     output_file: str):
    """Generate descriptions for consolidated objects."""
    
    # Create output dictionary with only consolidated objects
    consolidated_descriptions = {}
    
    # First, add existing descriptions for consolidated objects that already have them
    for obj in consolidated_objects:
        if obj in existing_descriptions:
            consolidated_descriptions[obj] = existing_descriptions[obj]
            logger.info(f"Reusing existing description for: {obj}")
    
    # Identify objects needing new descriptions
    objects_needing_descriptions = [obj for obj in consolidated_objects if obj not in consolidated_descriptions]
    
    logger.info(f"Need to generate {len(objects_needing_descriptions)} new descriptions")
    
    # Generate descriptions for objects that need them
    for i, obj in enumerate(objects_needing_descriptions, 1):
        logger.info(f"Generating description {i}/{len(objects_needing_descriptions)}: {obj}")
        
        description = generate_object_description(obj, api_key)
        if description:
            consolidated_descriptions[obj] = description
            logger.info(f"    ✓ Generated description")
        else:
            logger.error(f"    ✗ Failed to generate description")
        
        # Save progress every 10 objects
        if i % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(consolidated_descriptions, f, indent=2)
            logger.info(f"Saved progress: {i}/{len(objects_needing_descriptions)} descriptions")
        
        # Rate limiting
        import time
        time.sleep(0.5)
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(consolidated_descriptions, f, indent=2)
    
    logger.info(f"Generated descriptions for {len(objects_needing_descriptions)} objects")
    logger.info(f"Total descriptions saved: {len(consolidated_descriptions)}")
    logger.info(f"Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate descriptions for consolidated objects")
    parser.add_argument("--consolidated_objects", required=True, help="Path to consolidated objects file")
    parser.add_argument("--consolidation_mapping", required=True, help="Path to consolidation mapping JSON")
    parser.add_argument("--existing_descriptions", required=True, help="Path to existing descriptions JSON")
    parser.add_argument("--output_file", required=True, help="Output file for consolidated descriptions")
    parser.add_argument("--api_key", required=True, help="Gemini API key")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading consolidated objects...")
    consolidated_objects = load_consolidated_objects(args.consolidated_objects)
    
    logger.info("Loading consolidation mapping...")
    consolidation_mapping = load_consolidation_mapping(args.consolidation_mapping)
    
    logger.info("Loading existing descriptions...")
    existing_descriptions = load_existing_descriptions(args.existing_descriptions)
    
    # Generate descriptions
    generate_consolidated_descriptions(
        consolidated_objects,
        consolidation_mapping,
        existing_descriptions,
        args.api_key,
        args.output_file
    )

if __name__ == "__main__":
    main()
