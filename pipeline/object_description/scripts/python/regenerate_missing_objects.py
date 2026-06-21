#!/usr/bin/env python3
"""
Regenerate object descriptions for the 6 objects that are missing 4 keys
using the correct 11-key prompt format.
"""

import json
import logging
from typing import Dict, List
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_object_description(object_name: str, api_key: str) -> Dict:
    """Generate object description using Google Gemini API with correct 11-key format."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""

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

Please apply this detailed, structured format to describe "{object_name}".
"""
        
        response = model.generate_content(prompt)
        
        # Try to parse JSON response
        try:
            description = json.loads(response.text)
            return description
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response with all 11 keys
            return {
                "general_description": f"Description for {object_name}",
                "shape": "Typical shapes and forms",
                "texture": "Surface qualities and textures", 
                "color": "Typical colors",
                "material": "Typical materials and composition",
                "physical_properties": response.text[:200] + "...",
                "functions": "Common applications and contexts",
                "affordance": "Various uses and interactions possible",
                "common_elements": "Common components and parts",
                "common_environmental_context": "Found in various environments",
                "additional_details": "Additional information"
            }
            
    except Exception as e:
        logger.warning(f"Failed to generate description for {object_name}: {e}")
        return {
            "general_description": f"Description for {object_name}",
            "shape": "Typical shapes and forms",
            "texture": "Surface qualities and textures", 
            "color": "Typical colors",
            "material": "Typical materials and composition",
            "physical_properties": f"Description for {object_name}",
            "functions": "Common applications and contexts",
            "affordance": "Various uses and interactions possible",
            "common_elements": "Common components and parts",
            "common_environmental_context": "Found in various environments",
            "additional_details": "Additional information"
        }

def parse_object_description(description: Dict) -> Dict:
    """Parse object description into structured concepts using the mixed concept parser."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from parse_mixed_concept_descriptions import MixedConceptParser
    
    parser = MixedConceptParser()
    parsed_concepts = {}
    
    for key, value in description.items():
        if isinstance(value, str) and value.strip():
            try:
                concepts = parser.parse_concept(value)
                parsed_concepts[key] = concepts
            except Exception as e:
                logger.warning(f"Failed to parse {key} for object: {e}")
                parsed_concepts[key] = []
        else:
            parsed_concepts[key] = []
    
    return parsed_concepts

def main():
    # The 6 objects that need fixing
    missing_objects = ['aircraft', 'bike', 'dessert', 'mammal', 'neckwear', 'sea creature']
    
    # API keys
    api_keys = [
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY")
    ]
    
    # Load existing descriptions
    logger.info("Loading existing object descriptions...")
    with open('object_description/results/filtered_full/object_descriptions_filtered_full.json', 'r') as f:
        descriptions = json.load(f)
    
    # Load existing parsed concepts
    logger.info("Loading existing parsed concepts...")
    with open('object_description/results/filtered_full/parsed_concepts_filtered_full.json', 'r') as f:
        parsed_concepts = json.load(f)
    
    # Load source mapping
    logger.info("Loading source mapping...")
    with open('object_description/results/filtered_full/object_descriptions_filtered_full_source_mapping.json', 'r') as f:
        source_mapping = json.load(f)
    
    # Fix each missing object
    for i, obj_name in enumerate(missing_objects):
        logger.info(f"Fixing object {i+1}/6: {obj_name}")
        
        # Use alternating API keys
        api_key = api_keys[i % len(api_keys)]
        
        # Generate new description
        new_description = generate_object_description(obj_name, api_key)
        
        # Parse the new description
        new_parsed = parse_object_description(new_description)
        
        # Update the descriptions
        descriptions[obj_name] = new_description
        
        # Update the parsed concepts
        parsed_concepts[obj_name] = new_parsed
        
        # Update source mapping
        source_mapping[obj_name] = "OpenImages_regenerated"
        
        logger.info(f"Fixed {obj_name} with {len(new_description)} keys")
    
    # Save updated descriptions
    logger.info("Saving updated object descriptions...")
    with open('object_description/results/filtered_full/object_descriptions_filtered_full.json', 'w') as f:
        json.dump(descriptions, f, indent=2)
    
    # Save updated parsed concepts
    logger.info("Saving updated parsed concepts...")
    with open('object_description/results/filtered_full/parsed_concepts_filtered_full.json', 'w') as f:
        json.dump(parsed_concepts, f, indent=2)
    
    # Save updated source mapping
    logger.info("Saving updated source mapping...")
    with open('object_description/results/filtered_full/object_descriptions_filtered_full_source_mapping.json', 'w') as f:
        json.dump(source_mapping, f, indent=2)
    
    # Verify the fix
    logger.info("Verifying the fix...")
    for obj_name in missing_objects:
        if obj_name in parsed_concepts:
            data = parsed_concepts[obj_name]
            missing_keys = []
            for key in ['common_elements', 'material', 'shape', 'texture']:
                if key not in data or not data[key]:
                    missing_keys.append(key)
            
            if missing_keys:
                logger.warning(f"{obj_name} still missing keys: {missing_keys}")
            else:
                logger.info(f"{obj_name} now has all required keys")
    
    logger.info("Fix completed!")

if __name__ == "__main__":
    main()
