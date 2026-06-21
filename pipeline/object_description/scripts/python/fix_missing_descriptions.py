import os
#!/usr/bin/env python3
"""
Fix missing descriptions for objects with empty fields using the exact prompt format.
"""

import json
import google.generativeai as genai
import time
import sys
import re
from typing import Dict, Any

# API Configuration
API_KEY = os.environ.get("GEMINI_API_KEY")

def generate_prompt(category_name: str) -> str:
    """Generate the exact prompt format as specified"""
    return f'''

You are tasked with transforming the name of an object class into a structured description in JSON format. 
The goal is to represent each object class consistently with the same set of attributes, so that the output is machine-readable and uniform for clustering and machine learning tasks.

### Task
- Input: the name of a class of object (e.g., "wall", "building", "tree").
- Output: a JSON object where the class name is the key, and its value is another JSON object containing exactly **11 attributes**.
- **CRITICAL**: Each attribute value must be an **ARRAY of short, separate concept strings**.
- Every output must contain **all 11 keys** listed below, even if some values are empty.  
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
  "{category_name}": {{
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

Please describe "{category_name}" following this format. Keep each array item SHORT and CONCISE (5-10 words max per item).
'''

def extract_json_from_response(response_text: str, obj_name: str) -> Dict[str, Any]:
    """Extract JSON data from Gemini response"""
    # Clean up response
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]
    response_text = response_text.strip()
    
    # Try to find JSON content
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    for json_str in reversed(json_matches):  # Start with largest match
        try:
            parsed_data = json.loads(json_str)
            
            # Check if the object name is a key in the response
            if isinstance(parsed_data, dict) and obj_name in parsed_data:
                return parsed_data[obj_name]
            
            # If not, the response might be the object data directly
            if isinstance(parsed_data, dict) and all(key in parsed_data for key in [
                'general_description', 'shape', 'texture', 'color', 'material', 
                'physical_properties', 'functions', 'affordance', 'common_elements',
                'common_environmental_context', 'additional_details'
            ]):
                return parsed_data
                
        except json.JSONDecodeError:
            continue
    
    return None

def fix_empty_fields(descriptions: Dict[str, Dict], objects_to_fix: list) -> tuple:
    """Fix empty fields for specified objects"""
    
    # Configure Gemini
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    success_count = 0
    failed_objects = []
    
    print(f"Processing {len(objects_to_fix)} objects...")
    
    for i, obj_name in enumerate(objects_to_fix):
        print(f"\nProcessing {i+1}/{len(objects_to_fix)}: {obj_name}")
        
        try:
            # Generate description using the exact prompt format
            prompt = generate_prompt(obj_name)
            response = model.generate_content(prompt)
            
            if not response.text:
                print(f"  ✗ Empty response")
                failed_objects.append(obj_name)
                continue
                
            # Extract JSON from response
            new_data = extract_json_from_response(response.text, obj_name)
            
            if not new_data:
                print(f"  ✗ Could not extract JSON from response")
                failed_objects.append(obj_name)
                continue
            
            # Update only the empty fields
            updated = False
            original_desc = descriptions[obj_name]
            
            for field in [
                'physical_properties', 'material', 'affordance', 'functions',
                'additional_details', 'common_elements', 'common_environmental_context'
            ]:
                if field in original_desc:
                    current_value = original_desc[field]
                    
                    # Check if field is empty or has only empty strings
                    is_empty = False
                    if isinstance(current_value, list):
                        is_empty = len(current_value) == 0 or all(
                            item == '' or item is None for item in current_value
                        )
                    elif current_value == '' or current_value is None:
                        is_empty = True
                    
                    # Update if empty and we have new data
                    if is_empty and field in new_data and new_data[field]:
                        new_value = new_data[field]
                        
                        # Convert to proper list format to match existing structure
                        processed_items = []
                        
                        if isinstance(new_value, list):
                            # Already a list - clean and process each item
                            for item in new_value:
                                if isinstance(item, str) and item.strip():
                                    # Split comma-separated items (like fully_parse_concepts.py)
                                    if ',' in item:
                                        sub_items = [sub_item.strip() for sub_item in item.split(',') if sub_item.strip()]
                                        processed_items.extend(sub_items)
                                    else:
                                        processed_items.append(item.strip())
                        elif isinstance(new_value, str) and new_value.strip():
                            # String value - split by comma and clean up
                            if ',' in new_value:
                                # Split by comma and clean each item
                                processed_items = [item.strip() for item in new_value.split(',') if item.strip()]
                            else:
                                # Single value, put in list
                                processed_items = [new_value.strip()]
                        
                        if processed_items:
                            descriptions[obj_name][field] = processed_items
                            updated = True
            
            if updated:
                success_count += 1
                print(f"  ✓ Updated empty fields")
                
                # Show what was updated
                for field in ['physical_properties', 'material', 'affordance', 'functions']:
                    if field in descriptions[obj_name]:
                        value = descriptions[obj_name][field]
                        if isinstance(value, list) and len(value) > 0 and value[0] != '':
                            print(f"    {field}: {value[0][:50]}...")
            else:
                print(f"  ⚠ No fields were updated (may already be complete)")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_objects.append(obj_name)
        
        # Rate limiting
        time.sleep(1)
    
    return success_count, failed_objects

def main():
    """Main function to fix missing descriptions"""
    
    descriptions_file = '/path/to/SpatialReasonerDataGen/object_description/results/object_list_final/full_object_descriptions_fully_parsed.json'
    
    print("=" * 60)
    print("Fix Missing Descriptions in Object Descriptions File")
    print("=" * 60)
    
    # Load descriptions
    print("Loading object descriptions...")
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    
    print(f"Loaded {len(descriptions)} object descriptions")
    
    # Find objects with empty fields
    objects_with_empty_physical_props = []
    objects_with_other_empty_fields = []
    
    for obj_name, obj_desc in descriptions.items():
        # Check physical_properties
        if 'physical_properties' in obj_desc:
            props = obj_desc['physical_properties']
            if isinstance(props, list) and len(props) == 1 and props[0] == '':
                objects_with_empty_physical_props.append(obj_name)
        
        # Check other fields
        empty_fields = []
        for field in ['material', 'affordance', 'functions', 'additional_details', 'common_elements', 'common_environmental_context']:
            if field in obj_desc:
                value = obj_desc[field]
                if isinstance(value, list) and len(value) == 1 and value[0] == '':
                    empty_fields.append(field)
        
        if empty_fields and obj_name not in objects_with_empty_physical_props:
            objects_with_other_empty_fields.append(obj_name)
    
    print(f"\nFound {len(objects_with_empty_physical_props)} objects with empty physical_properties")
    print(f"Found {len(objects_with_other_empty_fields)} objects with other empty fields")
    
    # Fix objects with empty physical_properties first (these are the most important)
    if objects_with_empty_physical_props:
        print(f"\nFixing objects with empty physical_properties...")
        success_count, failed_objects = fix_empty_fields(descriptions, objects_with_empty_physical_props)
        
        print(f"\nResults:")
        print(f"  Successfully updated: {success_count}")
        print(f"  Failed: {len(failed_objects)}")
        
        if failed_objects:
            print(f"  Failed objects: {failed_objects[:5]}{'...' if len(failed_objects) > 5 else ''}")
    
    # Save updated file
    print(f"\nSaving updated descriptions file...")
    with open(descriptions_file, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
    
    print("✅ File updated successfully!")
    
    # Verification
    print("\nVerification - checking remaining empty fields...")
    remaining_empty_physical = 0
    for obj_name, obj_desc in descriptions.items():
        if 'physical_properties' in obj_desc:
            props = obj_desc['physical_properties']
            if isinstance(props, list) and len(props) == 1 and props[0] == '':
                remaining_empty_physical += 1
    
    print(f"Remaining objects with empty physical_properties: {remaining_empty_physical}")

if __name__ == "__main__":
    main()
