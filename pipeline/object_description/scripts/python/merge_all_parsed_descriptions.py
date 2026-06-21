#!/usr/bin/env python3

import json
import os
from pathlib import Path
from collections import defaultdict

def load_json_descriptions(file_path):
    descriptions = {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                for key, value in data.items():
                    obj_name = key.lower().strip()
                    if obj_name:
                        descriptions[obj_name] = value
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return descriptions

def merge_all_parsed_descriptions(results_dir):
    all_parsed_files = [
        "parsed/parsed_consolidated_descriptions.json",
        "parsed/parsed_object_descriptions_full.json",
        "parsed/parsed_SM_descriptions.json",
        "parsed/prolab_parsed/parsed_ade_descriptions.json",
        "parsed/prolab_parsed/parsed_city_bdd_descriptions.json",
        "parsed/prolab_parsed/parsed_coco_stuff_descriptions.json",
        "parsed/prolab_parsed/parsed_pascal_context_descriptions.json",
        "filtered_full/parsed_concepts_filtered_full.json",
    ]
    
    merged_parsed_concepts = {}
    merged_full_descriptions = {}
    source_tracking = defaultdict(list)
    
    print("Loading all parsed description files...")
    for rel_path in all_parsed_files:
        file_path = results_dir / rel_path
        if not file_path.exists():
            print(f"  ⚠️  Skipping {rel_path} (not found)")
            continue
        
        print(f"  📄 Loading {rel_path}...")
        data = load_json_descriptions(file_path)
        
        for obj_name, description in data.items():
            source_tracking[obj_name].append(rel_path)
            
            if obj_name not in merged_parsed_concepts:
                merged_parsed_concepts[obj_name] = description
            
            if "full_description" in str(description) or "description" in str(description):
                if obj_name not in merged_full_descriptions or len(str(description)) > len(str(merged_full_descriptions.get(obj_name, ""))):
                    merged_full_descriptions[obj_name] = description
        
        print(f"     → Loaded {len(data)} objects")
    
    return merged_parsed_concepts, merged_full_descriptions, dict(source_tracking)

def merge_with_object_list(merged_objects_file, parsed_descriptions):
    with open(merged_objects_file, 'r') as f:
        merged_objects = [line.strip().lower() for line in f if line.strip()]
    
    filtered_descriptions = {}
    for obj in merged_objects:
        if obj in parsed_descriptions:
            filtered_descriptions[obj] = parsed_descriptions[obj]
    
    return filtered_descriptions, merged_objects

def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    output_dir = results_dir / "full_object_description"
    
    print("=" * 80)
    print("MERGING ALL PARSED DESCRIPTIONS")
    print("=" * 80)
    
    merged_parsed_concepts, merged_full_descriptions, source_tracking = merge_all_parsed_descriptions(results_dir)
    
    print("\n" + "=" * 80)
    print(f"TOTAL UNIQUE OBJECTS FOUND: {len(merged_parsed_concepts)}")
    print("=" * 80)
    
    merged_objects_file = output_dir / "merged_objects_list.txt"
    
    print("\nFiltering to merged object list...")
    filtered_parsed, merged_objects = merge_with_object_list(merged_objects_file, merged_parsed_concepts)
    
    output_parsed = output_dir / "parsed_concepts_full_merged_complete.json"
    with open(output_parsed, 'w') as f:
        json.dump(dict(sorted(filtered_parsed.items())), f, indent=2)
    print(f"✅ Saved: {output_parsed}")
    print(f"   Objects with descriptions: {len(filtered_parsed)}/{len(merged_objects)}")
    
    filtered_full, _ = merge_with_object_list(merged_objects_file, merged_full_descriptions)
    output_full = output_dir / "full_object_descriptions_merged_complete.json"
    with open(output_full, 'w') as f:
        json.dump(dict(sorted(filtered_full.items())), f, indent=2)
    print(f"✅ Saved: {output_full}")
    print(f"   Objects with full descriptions: {len(filtered_full)}/{len(merged_objects)}")
    
    missing_objects = [obj for obj in merged_objects if obj not in merged_parsed_concepts]
    if missing_objects:
        missing_file = output_dir / "objects_without_descriptions.txt"
        with open(missing_file, 'w') as f:
            for obj in sorted(missing_objects):
                f.write(f"{obj}\n")
        print(f"\n⚠️  {len(missing_objects)} objects have no descriptions")
        print(f"   List saved to: {missing_file}")
        print(f"   Sample missing: {', '.join(missing_objects[:10])}")
    
    source_map_file = output_dir / "description_sources.json"
    with open(source_map_file, 'w') as f:
        json.dump(dict(sorted(source_tracking.items())), f, indent=2)
    print(f"\n✅ Source tracking saved: {source_map_file}")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\n📊 FINAL STATS:")
    print(f"   Total objects in merged list: {len(merged_objects)}")
    print(f"   Objects with parsed concepts: {len(filtered_parsed)} ({len(filtered_parsed)/len(merged_objects)*100:.1f}%)")
    print(f"   Objects with full descriptions: {len(filtered_full)} ({len(filtered_full)/len(merged_objects)*100:.1f}%)")
    print(f"   Objects without descriptions: {len(missing_objects)} ({len(missing_objects)/len(merged_objects)*100:.1f}%)")

if __name__ == "__main__":
    main()

