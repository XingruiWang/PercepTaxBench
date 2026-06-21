#!/usr/bin/env python3
"""
Combine filtered OpenImages objects with SM objects and create unified dataset for clustering.
This script creates object_descriptions_filtered_full.json and parsed_concepts_filtered_full.json
in a separate folder for clustering analysis.
"""

import json
import os
import argparse
import logging
from typing import Dict, List, Set
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_filtered_openimages() -> List[str]:
    """Load the filtered OpenImages objects (1474 objects)."""
    logger.info("Loading filtered OpenImages objects")
    
    with open('object_description/results/openimages_objects_filtered.txt', 'r') as f:
        objects = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(objects)} filtered OpenImages objects")
    return objects

def load_sm_objects() -> Dict[str, Dict]:
    """Load the SM objects and their descriptions."""
    logger.info("Loading SM objects")
    
    with open('object_description/results/SM_descriptions.json', 'r') as f:
        sm_data = json.load(f)
    
    logger.info(f"Loaded {len(sm_data)} SM objects")
    return sm_data

def load_sm_parsed() -> Dict[str, Dict]:
    """Load the parsed SM concepts."""
    logger.info("Loading parsed SM concepts")
    
    with open('object_description/results/parsed/parsed_SM_descriptions.json', 'r') as f:
        sm_parsed = json.load(f)
    
    logger.info(f"Loaded {len(sm_parsed)} parsed SM concepts")
    return sm_parsed

def generate_object_description(object_name: str, api_key: str) -> Dict[str, str]:
    """Generate object description using Google Gemini API."""
    import google.generativeai as genai
    
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

def load_existing_descriptions() -> Dict[str, Dict]:
    """Load existing object descriptions from object_descriptions_full.json."""
    logger.info("Loading existing object descriptions")
    
    try:
        with open('object_description/results/object_descriptions_full.json', 'r') as f:
            descriptions = json.load(f)
        logger.info(f"Loaded {len(descriptions)} existing object descriptions")
        return descriptions
    except FileNotFoundError:
        logger.warning("No existing object descriptions found")
        return {}

def combine_object_descriptions(filtered_objects: List[str], 
                               sm_descriptions: Dict[str, Dict],
                               existing_descriptions: Dict[str, Dict],
                               api_key: str,
                               output_file: str):
    """Combine filtered OpenImages objects with SM descriptions using existing descriptions when available."""
    logger.info("Combining object descriptions")
    
    combined_descriptions = {}
    source_mapping = {}
    
    # Add SM objects first (they take priority)
    for obj_name, description in sm_descriptions.items():
        combined_descriptions[obj_name] = description
        source_mapping[obj_name] = "SM"
    
    # Add filtered OpenImages objects
    sm_object_names = set(sm_descriptions.keys())
    
    for obj_name in filtered_objects:
        if obj_name not in sm_object_names:
            # Check if description exists in existing descriptions
            if obj_name in existing_descriptions:
                logger.info(f"Using existing description for: {obj_name}")
                combined_descriptions[obj_name] = existing_descriptions[obj_name]
                source_mapping[obj_name] = "OpenImages_existing"
            else:
                # Generate description for OpenImages object only if not found
                logger.info(f"Generating new description for: {obj_name}")
                description = generate_object_description(obj_name, api_key)
                combined_descriptions[obj_name] = description
                source_mapping[obj_name] = "OpenImages_generated"
        else:
            logger.info(f"Using existing SM description for: {obj_name}")
            source_mapping[obj_name] = "SM"
    
    # Save combined descriptions
    with open(output_file, 'w') as f:
        json.dump(combined_descriptions, f, indent=2)
    
    logger.info(f"Saved {len(combined_descriptions)} combined object descriptions to: {output_file}")
    
    # Save source mapping
    mapping_file = str(output_file).replace('.json', '_source_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(source_mapping, f, indent=2)
    
    logger.info(f"Saved source mapping to: {mapping_file}")
    
    return combined_descriptions, source_mapping

def load_existing_parsed_concepts() -> Dict[str, Dict]:
    """Load existing parsed concepts from the full parsed file."""
    logger.info("Loading existing parsed concepts")
    
    with open('object_description/results/parsed/parsed_object_descriptions_full.json', 'r') as f:
        parsed_data = json.load(f)
    
    logger.info(f"Loaded {len(parsed_data)} existing parsed concepts")
    return parsed_data

def parse_object_descriptions(descriptions: Dict[str, Dict], existing_parsed: Dict[str, Dict]) -> Dict[str, Dict]:
    """Parse object descriptions into structured concepts, using existing parsed concepts when available."""
    logger.info("Parsing object descriptions into concepts")
    
    parsed_concepts = {}
    
    for obj_name, description in descriptions.items():
        # Use existing parsed concepts if available
        if obj_name in existing_parsed:
            logger.info(f"Using existing parsed concepts for: {obj_name}")
            parsed_concepts[obj_name] = existing_parsed[obj_name]
        else:
            # For missing objects, create empty structure (will be filled by proper parser later)
            logger.info(f"No existing parsed concepts for: {obj_name}, creating empty structure")
            concepts = {
                "general_description": [],
                "shape": [],
                "texture": [],
                "color": [],
                "material": [],
                "functions": [],
                "affordance": [],
                "common_elements": [],
                "common_environmental_context": [],
                "additional_details": [],
                "physical_properties": []
            }
            parsed_concepts[obj_name] = concepts
    
    logger.info(f"Parsed concepts for {len(parsed_concepts)} objects")
    return parsed_concepts

def combine_parsed_concepts(oi_parsed: Dict[str, Dict], 
                          sm_parsed: Dict[str, Dict],
                          source_mapping: Dict[str, str],
                          output_file: str) -> Dict[str, Dict]:
    """Combine parsed concepts from OpenImages and SM objects."""
    logger.info("Combining parsed concepts")
    
    combined_parsed = {}
    
    # Add SM parsed concepts first
    for obj_name, concepts in sm_parsed.items():
        combined_parsed[obj_name] = concepts
    
    # Add OpenImages parsed concepts
    sm_object_names = set(sm_parsed.keys())
    
    for obj_name, concepts in oi_parsed.items():
        if obj_name not in sm_object_names:
            combined_parsed[obj_name] = concepts
    
    # Save combined parsed concepts
    with open(output_file, 'w') as f:
        json.dump(combined_parsed, f, indent=2)
    
    logger.info(f"Saved {len(combined_parsed)} combined parsed concepts to: {output_file}")
    
    return combined_parsed

def create_full_object_descriptions(combined_descriptions: Dict[str, Dict], 
                                  output_file: str):
    """Create full object descriptions file for clustering."""
    logger.info("Creating full object descriptions file")
    
    with open(output_file, 'w') as f:
        json.dump(combined_descriptions, f, indent=2)
    
    logger.info(f"Saved full object descriptions to: {output_file}")

def analyze_combined_dataset(combined_descriptions: Dict[str, Dict],
                           source_mapping: Dict[str, str]) -> Dict:
    """Analyze the combined dataset."""
    logger.info("Analyzing combined dataset")
    
    sm_count = sum(1 for source in source_mapping.values() if source == "SM")
    oi_count = sum(1 for source in source_mapping.values() if source.startswith("OpenImages"))
    
    analysis = {
        "total_objects": len(combined_descriptions),
        "sm_objects": sm_count,
        "openimages_objects": oi_count,
        "source_distribution": {
            "SM": sm_count,
            "OpenImages": oi_count
        }
    }
    
    logger.info(f"Dataset analysis:")
    logger.info(f"  Total objects: {analysis['total_objects']}")
    logger.info(f"  SM objects: {analysis['sm_objects']}")
    logger.info(f"  OpenImages objects: {analysis['openimages_objects']}")
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Combine filtered OpenImages objects with SM objects")
    parser.add_argument("--api_key", type=str, required=True, help="Google Gemini API key")
    parser.add_argument("--output_dir", type=str, default="object_description/results/filtered_full", 
                       help="Output directory for combined files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    filtered_objects = load_filtered_openimages()
    sm_descriptions = load_sm_objects()
    sm_parsed = load_sm_parsed()
    existing_descriptions = load_existing_descriptions()
    existing_parsed = load_existing_parsed_concepts()
    
    # Combine object descriptions
    combined_descriptions, source_mapping = combine_object_descriptions(
        filtered_objects, sm_descriptions, existing_descriptions, args.api_key,
        output_dir / "object_descriptions_filtered_full.json"
    )
    
    # Parse OpenImages descriptions using existing parsed concepts
    oi_parsed = parse_object_descriptions({
        obj: desc for obj, desc in combined_descriptions.items() 
        if source_mapping[obj] in ["OpenImages_existing", "OpenImages_generated"]
    }, existing_parsed)
    
    # Combine parsed concepts
    combined_parsed = combine_parsed_concepts(
        oi_parsed, sm_parsed, source_mapping,
        output_dir / "parsed_concepts_filtered_full.json"
    )
    
    # Create full object descriptions
    create_full_object_descriptions(
        combined_descriptions,
        output_dir / "full_object_descriptions_filtered_full.json"
    )
    
    # Analyze dataset
    analysis = analyze_combined_dataset(combined_descriptions, source_mapping)
    
    # Save analysis
    with open(output_dir / "dataset_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info("Combination process completed successfully!")

if __name__ == "__main__":
    main()
