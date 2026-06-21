#!/usr/bin/env python3
"""
Generate descriptions for missing OpenImages objects

This script identifies objects detected in OpenImages that don't have descriptions
yet and generates descriptions for them using the Gemini API.
"""

import json
import os
import time
from typing import Dict, Any

# API Keys for rotation
API_KEYS = [
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY")
]

def generate_object_description(object_name: str, api_key: str) -> Dict[str, str]:
    """Generate comprehensive object description using Gemini API."""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""Generate a comprehensive description for the object "{object_name}". 

Provide detailed information covering these aspects:
- General description: What is this object?
- Shape: What shape/form does it typically have?
- Texture: What does it feel like to touch?
- Color: What colors is it commonly found in?
- Material: What is it typically made of?
- Functions: What is its primary purpose or use?
- Affordance: How do people typically interact with it?
- Common elements: What parts or components does it usually have?
- Common environmental context: Where is it typically found?
- Additional details: Any other relevant characteristics
- Physical properties: Size, weight, durability, etc.

Format your response as a JSON object with these exact keys:
{{
    "general_description": "...",
    "shape": "...",
    "texture": "...",
    "color": "...",
    "material": "...",
    "functions": "...",
    "affordance": "...",
    "common_elements": "...",
    "common_environmental_context": "...",
    "additional_details": "...",
    "physical_properties": "..."
}}

Make sure each field contains meaningful, detailed information about the object."""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response if it has markdown formatting
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON response
        description = json.loads(response_text)
        
        # Validate that all required keys are present
        required_keys = [
            'general_description', 'shape', 'texture', 'color', 'material',
            'functions', 'affordance', 'common_elements', 'common_environmental_context',
            'additional_details', 'physical_properties'
        ]
        
        for key in required_keys:
            if key not in description:
                description[key] = f"Information about {object_name} {key.replace('_', ' ')}"
        
        return description
        
    except Exception as e:
        print(f"Error generating description for {object_name}: {e}")
        return None

def main():
    """Main function to generate descriptions for missing objects."""
    
    print("=" * 60)
    print("Generate Descriptions for Missing OpenImages Objects")
    print("=" * 60)
    
    # Load detected objects
    print("Loading detected objects...")
    with open('openimages_3d_annotations/data/openimages_detected_objects.txt', 'r') as f:
        detected_objects = set(line.strip() for line in f)
    
    # Load existing descriptions
    print("Loading existing descriptions...")
    with open('object_description/results/object_descriptions_full.json', 'r') as f:
        existing_descriptions = json.load(f)
    
    # Find missing objects
    missing_objects = detected_objects - set(existing_descriptions.keys())
    
    print(f"\nFound {len(missing_objects)} missing objects:")
    for obj in sorted(missing_objects):
        print(f"  - {obj}")
    
    if not missing_objects:
        print("\n✓ All detected objects already have descriptions!")
        return
    
    # Generate descriptions for missing objects
    print(f"\nGenerating descriptions for {len(missing_objects)} missing objects...")
    
    new_descriptions = {}
    total_processed = 0
    total_errors = 0
    
    for i, obj_name in enumerate(sorted(missing_objects)):
        print(f"\nProcessing {i+1}/{len(missing_objects)}: {obj_name}")
        
        # Rotate API keys
        api_key = API_KEYS[total_processed % len(API_KEYS)]
        
        description = generate_object_description(obj_name, api_key)
        
        if description:
            new_descriptions[obj_name] = description
            total_processed += 1
            print(f"  ✓ Generated description")
        else:
            total_errors += 1
            print(f"  ✗ Failed to generate description")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Merge with existing descriptions
    print(f"\nMerging with existing descriptions...")
    all_descriptions = {**existing_descriptions, **new_descriptions}
    
    # Save updated descriptions
    print(f"Saving updated descriptions...")
    with open('object_description/results/object_descriptions_full.json', 'w') as f:
        json.dump(all_descriptions, f, indent=2)
    
    print(f"\n✓ Completed!")
    print(f"  New descriptions generated: {total_processed}")
    print(f"  Errors: {total_errors}")
    print(f"  Total objects in database: {len(all_descriptions)}")
    print(f"  Updated file: object_description/results/object_descriptions_full.json")

if __name__ == "__main__":
    main()
