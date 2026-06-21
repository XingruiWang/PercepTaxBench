#!/usr/bin/env python3
"""
Generate descriptions for objects missing from full_object_descriptions_fully_parsed.json

Uses Gemini API to generate structured descriptions with taxonomy clusters.
"""

import json
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any
import google.generativeai as genai

API_KEYS = [
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY")
]

def validate_and_fix_arrays(description: Dict[str, Any]) -> Dict[str, list]:
    """Ensure all values are arrays, convert strings to single-item arrays if needed"""
    fixed = {}
    
    for key, value in description.items():
        if isinstance(value, list):
            # Already a list, keep it
            fixed[key] = value
        elif isinstance(value, str):
            # Convert string to single-item list
            fixed[key] = [value.strip()] if value.strip() else []
        else:
            # Other types, wrap in list
            fixed[key] = [value] if value else []
    
    return fixed


def generate_object_description(object_name: str, api_key: str) -> Dict[str, Any]:
    """Generate comprehensive object description using the same format as original script."""
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
You are tasked with transforming the name of an object class into a structured description in JSON format. 
The goal is to represent each object class consistently with the same set of attributes, so that the output is machine-readable and uniform for clustering and machine learning tasks.

### Task
- Input: the name of a class of object (e.g., "wall", "building", "tree").
- Output: a JSON object where the class name is the key, and its value is another JSON object containing exactly **11 attributes**.
- **CRITICAL**: Each attribute value must be an **ARRAY of short, separate concept strings**.
- Keep descriptions CONCISE - 2-5 short phrases per attribute, NOT long paragraphs.
- Break down descriptions into brief, atomic concepts as separate strings in the array.
- The JSON must be syntactically valid.

### Required 11 keys (all must be ARRAYS of SHORT strings):
1. "general_description" – array with 1-2 brief sentences (e.g., ["A tool for cutting", "Used in offices and homes"])
2. "shape" – array of brief shape descriptors (e.g., ["rectangular", "curved edges", "box-like"])
3. "texture" – array of brief texture concepts (e.g., ["smooth", "rough surface", "textured grip"])
4. "color" – array of typical colors (e.g., ["black", "white", "silver"])
5. "material" – array of materials (e.g., ["plastic", "metal", "rubber"])
6. "physical_properties" – array of properties, choose ONE from each pair: [heavy/light, rigid/flexible, durable/fragile, smooth/rough, movable/fixed, stable/unstable, solid/liquid/gas, solid/hollow]
   (e.g., ["heavy", "rigid", "durable", "smooth", "movable"]) DO NOT choose two words from the same pair.
7. "functions" – array of brief uses (e.g., ["cuts paper", "trims fabric", "craft projects"])
8. "affordance" – array of brief interaction methods (e.g., ["grasp handles", "squeeze blades", "position item"])
9. "common_elements" – array of associated objects (e.g., ["desk", "paper", "holder"])
10. "common_environmental_context" – array of brief locations (e.g., ["office", "classroom", "home"])
11. "additional_details" – array of brief variations (e.g., ["left-handed versions", "safety scissors", "decorative"])

### Example format:
{{
  "{object_name}": {{
    "general_description": ["Brief description of the object", "What it is used for"],
    "shape": ["descriptor 1", "descriptor 2"],
    "texture": ["texture 1", "texture 2"],
    "color": ["color 1", "color 2"],
    "material": ["material 1", "material 2"],
    "physical_properties": ["heavy", "rigid", "durable", "smooth", "movable"],
    "functions": ["function 1", "function 2"],
    "affordance": ["interaction 1", "interaction 2"],
    "common_elements": ["element 1", "element 2"],
    "common_environmental_context": ["context 1", "context 2"],
    "additional_details": ["detail 1", "detail 2"]
  }}
}}

Please describe "{object_name}" following this format. Keep each array item SHORT and CONCISE (5-10 words max per item). Generate decent detail but NOT long paragraphs.
"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        parsed = json.loads(response_text)
        
        # The response should be {object_name: {attributes}}
        # Extract the description dict
        if object_name in parsed:
            description = parsed[object_name]
        elif len(parsed) == 1:
            # If only one key, assume it's the object description
            description = list(parsed.values())[0]
        else:
            # Assume the parsed dict is already the description
            description = parsed
        
        # Validate required keys
        required_keys = [
            'general_description', 'shape', 'texture', 'color', 'material',
            'functions', 'affordance', 'common_elements', 'common_environmental_context',
            'additional_details', 'physical_properties'
        ]
        
        for key in required_keys:
            if key not in description:
                description[key] = [f"Information about {object_name} {key.replace('_', ' ')}"]
        
        # Ensure all values are arrays (in case Gemini returned some as strings)
        description = validate_and_fix_arrays(description)
        
        return description
        
    except Exception as e:
        print(f"  Error generating description for {object_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate descriptions for missing objects')
    parser.add_argument('--max-objects', type=int, default=None, help='Maximum number of objects to process')
    parser.add_argument('--checkpoint-file', type=str, default=None, help='Checkpoint file for resume capability')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Generate Descriptions for Missing Objects")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results" / "full_object_description"
    
    # Output and checkpoint paths
    output_path = results_dir / "new_object_descriptions.json"
    full_descriptions_path = results_dir / "full_object_descriptions_fully_parsed.json"
    if args.checkpoint_file:
        checkpoint_path = Path(args.checkpoint_file)
    else:
        checkpoint_path = results_dir / "description_generation_checkpoint.json"
    
    # Load checkpoint if exists
    new_descriptions = {}
    processed_objects = set()
    if checkpoint_path.exists():
        print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            new_descriptions = checkpoint_data.get('descriptions', {})
            processed_objects = set(checkpoint_data.get('processed', []))
        print(f"   Resuming: {len(new_descriptions)} objects already processed")
    
    # Load missing objects list - use CORRECT file based on all_objects_merged.txt
    missing_path_correct = results_dir / "objects_missing_descriptions_CORRECT.txt"
    missing_path_old = results_dir / "objects_missing_descriptions.txt"
    
    if missing_path_correct.exists():
        missing_path = missing_path_correct
        print(f"✅ Using CORRECT missing list: {missing_path_correct}")
    else:
        missing_path = missing_path_old
        print(f"⚠️  Using old missing list: {missing_path_old}")
    
    with open(missing_path, 'r') as f:
        all_missing_objects = [line.strip() for line in f if line.strip()]
    
    # Filter out already processed objects
    missing_objects = [obj for obj in all_missing_objects if obj not in processed_objects]
    
    # Load existing full descriptions - THIS IS THE MASTER FILE THAT GETS UPDATED
    with open(full_descriptions_path, 'r') as f:
        existing_descriptions = json.load(f)
    
    print(f"\n📚 Loaded master description file with {len(existing_descriptions)} objects")
    
    if args.max_objects:
        missing_objects = missing_objects[:args.max_objects]
    
    print(f"\nTotal missing objects: {len(all_missing_objects)}")
    print(f"Already processed: {len(processed_objects)}")
    print(f"Remaining to process: {len(missing_objects)}")
    print(f"Existing descriptions: {len(existing_descriptions)}")
    
    api_key_index = 0
    failed_objects = []
    
    for i, obj_name in enumerate(missing_objects, 1):
        print(f"\n[{len(processed_objects) + i}/{len(all_missing_objects)}] Processing: {obj_name}")
        
        api_key = API_KEYS[api_key_index]
        description = generate_object_description(obj_name, api_key)
        
        if description:
            new_descriptions[obj_name] = description
            processed_objects.add(obj_name)
            
            # IMMEDIATELY merge into the existing descriptions
            existing_descriptions[obj_name] = description
            print(f"  ✅ Generated and merged into master file")
            
            # Save checkpoint every 10 objects
            if i % 10 == 0:
                checkpoint_data = {
                    'descriptions': new_descriptions,
                    'processed': list(processed_objects)
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                # Also save the UPDATED FULL descriptions file
                with open(full_descriptions_path, 'w') as f:
                    json.dump(existing_descriptions, f, indent=2)
                
                print(f"  💾 Checkpoint + Full descriptions saved ({len(existing_descriptions)} total)")
        else:
            failed_objects.append(obj_name)
            print(f"  ❌ Failed")
        
        api_key_index = (api_key_index + 1) % len(API_KEYS)
        time.sleep(1)
    
    # Save final outputs
    with open(output_path, 'w') as f:
        json.dump(new_descriptions, f, indent=2)
    
    # Save final checkpoint
    checkpoint_data = {
        'descriptions': new_descriptions,
        'processed': list(processed_objects)
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    # Save FINAL UPDATED full descriptions file
    with open(full_descriptions_path, 'w') as f:
        json.dump(existing_descriptions, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total missing: {len(all_missing_objects)}")
    print(f"Processed this run: {len(missing_objects)}")
    print(f"Success: {len(new_descriptions)}")
    print(f"Failed: {len(failed_objects)}")
    print(f"\n📁 New descriptions: {output_path}")
    print(f"📚 MASTER file updated: {full_descriptions_path} ({len(existing_descriptions)} total objects)")
    print(f"💾 Checkpoint: {checkpoint_path}")
    
    if failed_objects:
        print(f"\n❌ Failed objects:")
        for obj in failed_objects:
            print(f"  - {obj}")


if __name__ == "__main__":
    main()

