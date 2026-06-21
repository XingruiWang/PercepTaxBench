#!/usr/bin/env python3
"""
Generate object descriptions for OpenImages detected objects
and update the object_descriptions_full.json file.
"""

import json
import os
import time
import random
from pathlib import Path
import google.generativeai as genai

# API keys for rotation
API_KEYS = [
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY"),  # Reuse first key as backup
]

def load_existing_descriptions():
    """Load existing object descriptions."""
    descriptions_file = "object_description/results/object_descriptions_full.json"
    
    if os.path.exists(descriptions_file):
        with open(descriptions_file, 'r') as f:
            return json.load(f)
    else:
        return {}

def load_openimages_objects():
    """Load the list of OpenImages detected objects."""
    with open("openimages_3d_annotations/data/openimages_detected_objects.txt", 'r') as f:
        return [line.strip() for line in f if line.strip()]

def generate_object_description(object_name, api_key):
    """Generate description for a single object using Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
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
                print(f"Could not parse JSON for {object_name}: {response_text[:200]}...")
                return None
                
    except Exception as e:
        print(f"Error generating description for {object_name}: {e}")
        return None

def main():
    # Load existing descriptions
    print("Loading existing object descriptions...")
    all_descriptions = load_existing_descriptions()
    print(f"Found {len(all_descriptions)} existing descriptions")
    
    # Load OpenImages objects
    print("Loading OpenImages detected objects...")
    openimages_objects = load_openimages_objects()
    print(f"Found {len(openimages_objects)} OpenImages objects")
    
    # Filter out objects that already have descriptions
    objects_to_process = []
    for obj in openimages_objects:
        if obj not in all_descriptions:
            objects_to_process.append(obj)
    
    print(f"Need to generate descriptions for {len(objects_to_process)} new objects")
    
    if not objects_to_process:
        print("All OpenImages objects already have descriptions!")
        return
    
    # Process objects in batches
    batch_size = 50
    total_processed = 0
    total_errors = 0
    
    for i in range(0, len(objects_to_process), batch_size):
        batch = objects_to_process[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(objects_to_process)-1)//batch_size + 1}")
        print(f"Objects: {batch[0]} to {batch[-1]}")
        
        for j, obj_name in enumerate(batch):
            print(f"  Processing {j+1}/{len(batch)}: {obj_name}")
            
            # Rotate API keys
            api_key = API_KEYS[total_processed % len(API_KEYS)]
            
            description = generate_object_description(obj_name, api_key)
            
            if description:
                all_descriptions[obj_name] = description
                total_processed += 1
                print(f"    ✓ Generated description")
            else:
                total_errors += 1
                print(f"    ✗ Failed to generate description")
            
            # Rate limiting
            time.sleep(0.5)
        
        # Save progress after each batch
        print(f"Saving progress... ({total_processed} descriptions generated)")
        with open("object_description/results/object_descriptions_full.json", 'w') as f:
            json.dump(all_descriptions, f, indent=2)
    
    print(f"\nCompleted!")
    print(f"Total descriptions generated: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Total objects in database: {len(all_descriptions)}")
    
    # Save final results
    with open("object_descriptions_output/object_descriptions_full.json", 'w') as f:
        json.dump(all_descriptions, f, indent=2)
    
    print("Updated object_descriptions_full.json with OpenImages objects!")

if __name__ == "__main__":
    main()
