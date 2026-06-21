#!/usr/bin/env python3

import json
import os
from pathlib import Path

def merge_object_lists(sm_file, openimages_file):
    objects = set()
    
    with open(sm_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                objects.add(line.lower())
    
    with open(openimages_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                objects.add(line.lower())
    
    return sorted(list(objects))

def filter_descriptions(merged_objects, input_json_file, output_json_file):
    merged_objects_set = set(merged_objects)
    
    with open(input_json_file, 'r') as f:
        all_descriptions = json.load(f)
    
    filtered_descriptions = {}
    filtered_count = 0
    
    for obj_name, description in all_descriptions.items():
        if obj_name.lower() in merged_objects_set:
            filtered_descriptions[obj_name] = description
            filtered_count += 1
    
    with open(output_json_file, 'w') as f:
        json.dump(filtered_descriptions, f, indent=2)
    
    print(f"Total objects in merged list: {len(merged_objects)}")
    print(f"Total descriptions in input file: {len(all_descriptions)}")
    print(f"Filtered descriptions: {filtered_count}")
    print(f"Descriptions not found: {len(merged_objects) - filtered_count}")
    
    return filtered_descriptions

def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    
    sm_file = results_dir / "sm_objects_138.txt"
    openimages_file = results_dir / "original_openimages_objects.txt"
    
    print("Merging object lists...")
    merged_objects = merge_object_lists(sm_file, openimages_file)
    
    output_dir = results_dir / "full_object_description"
    output_dir.mkdir(exist_ok=True)
    
    merged_objects_file = output_dir / "merged_objects_list.txt"
    with open(merged_objects_file, 'w') as f:
        for obj in merged_objects:
            f.write(f"{obj}\n")
    print(f"Saved merged object list to: {merged_objects_file}")
    print(f"Total unique objects: {len(merged_objects)}")
    
    print("\nFiltering descriptions...")
    input_json = results_dir / "filtered_full" / "parsed_concepts_filtered_full.json"
    output_json = output_dir / "parsed_concepts_full_merged.json"
    
    filtered_descriptions = filter_descriptions(merged_objects, input_json, output_json)
    print(f"Saved filtered descriptions to: {output_json}")
    
    output_json_full_desc = output_dir / "full_object_descriptions_merged.json"
    input_json_full_desc = results_dir / "filtered_full" / "full_object_descriptions_filtered_full.json"
    
    if input_json_full_desc.exists():
        print("\nFiltering full object descriptions...")
        filtered_full_descriptions = filter_descriptions(merged_objects, input_json_full_desc, output_json_full_desc)
        print(f"Saved filtered full descriptions to: {output_json_full_desc}")

if __name__ == "__main__":
    main()

