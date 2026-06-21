#!/usr/bin/env python3

import json
import re
from pathlib import Path

KEYS_TO_SPLIT = [
    'physical_properties',
    'common_elements',
    'material',
    'color',
    'texture',
    'functions',
]

def parse_comma_separated_values(text):
    if not isinstance(text, str):
        return [text]
    
    items = [item.strip() for item in text.split(',') if item.strip()]
    
    return items if len(items) > 1 else [text]

def fully_parse_description(description):
    if not isinstance(description, dict):
        return description
    
    parsed = {}
    
    for key, value in description.items():
        if not isinstance(value, list):
            parsed[key] = value
            continue
        
        parsed_values = []
        for item in value:
            if isinstance(item, str) and key in KEYS_TO_SPLIT:
                sub_items = parse_comma_separated_values(item)
                parsed_values.extend(sub_items)
            else:
                parsed_values.append(item)
        
        parsed[key] = parsed_values
    
    return parsed

def fully_parse_all_descriptions(input_file, output_file):
    print(f"\nProcessing: {input_file.name}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total objects: {len(data)}")
    
    parsed_data = {}
    changes_made = 0
    
    for obj_name, description in data.items():
        original_desc = json.dumps(description, sort_keys=True)
        parsed_desc = fully_parse_description(description)
        new_desc = json.dumps(parsed_desc, sort_keys=True)
        
        if original_desc != new_desc:
            changes_made += 1
        
        parsed_data[obj_name] = parsed_desc
    
    with open(output_file, 'w') as f:
        json.dump(parsed_data, f, indent=2)
    
    print(f"✅ Saved to: {output_file.name}")
    print(f"   Objects modified: {changes_made}/{len(data)}")
    
    sample_obj = list(parsed_data.items())[0]
    print(f"\n📋 Sample object: {sample_obj[0]}")
    if 'physical_properties' in sample_obj[1]:
        props = sample_obj[1]['physical_properties']
        print(f"   physical_properties: {len(props)} items")
        print(f"   → {props[:5]}...")
    
    return parsed_data

def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "results" / "full_object_description"
    
    print("="*80)
    print("FULLY PARSING ALL COMMA-SEPARATED VALUES")
    print("="*80)
    
    input_file = output_dir / "parsed_concepts_full_merged_complete.json"
    output_file = output_dir / "parsed_concepts_fully_parsed.json"
    
    parsed_data = fully_parse_all_descriptions(input_file, output_file)
    
    input_file2 = output_dir / "full_object_descriptions_merged_complete.json"
    output_file2 = output_dir / "full_object_descriptions_fully_parsed.json"
    
    parsed_data2 = fully_parse_all_descriptions(input_file2, output_file2)
    
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    print("\nChecking yoga mat example:")
    if 'yoga mat' in parsed_data:
        yoga_mat = parsed_data['yoga mat']
        if 'physical_properties' in yoga_mat:
            props = yoga_mat['physical_properties']
            print(f"  physical_properties ({len(props)} items):")
            for i, prop in enumerate(props, 1):
                print(f"    {i}. {prop}")
    
    print("\nChecking zebra example:")
    if 'zebra' in parsed_data:
        zebra = parsed_data['zebra']
        if 'material' in zebra:
            materials = zebra['material']
            print(f"  material ({len(materials)} items):")
            for i, mat in enumerate(materials, 1):
                print(f"    {i}. {mat}")
    
    print("\n" + "="*80)
    print("✅ COMPLETE! All comma-separated values have been parsed.")
    print("="*80)
    print(f"\n📁 Output files:")
    print(f"   1. parsed_concepts_fully_parsed.json")
    print(f"   2. full_object_descriptions_fully_parsed.json")

if __name__ == "__main__":
    main()

