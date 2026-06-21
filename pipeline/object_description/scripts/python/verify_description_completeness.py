#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict

EXPECTED_KEYS = [
    'general_description',
    'material',
    'shape',
    'color',
    'texture',
    'physical_properties',
    'affordance',
    'functions',
    'common_elements',
    'common_environmental_context',
    'additional_details'
]

def verify_description_completeness(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"VERIFYING: {json_file.name}")
    print(f"{'='*80}\n")
    
    total_objects = len(data)
    objects_with_all_keys = 0
    objects_missing_keys = []
    key_coverage = defaultdict(int)
    empty_values = defaultdict(list)
    
    for obj_name, description in data.items():
        if not isinstance(description, dict):
            print(f"❌ {obj_name}: Description is not a dictionary! Type: {type(description)}")
            objects_missing_keys.append((obj_name, "Not a dict"))
            continue
        
        present_keys = set(description.keys())
        expected_keys_set = set(EXPECTED_KEYS)
        
        missing_keys = expected_keys_set - present_keys
        
        for key in EXPECTED_KEYS:
            if key in description:
                key_coverage[key] += 1
                
                value = description[key]
                if not value or (isinstance(value, list) and len(value) == 0):
                    empty_values[key].append(obj_name)
            else:
                empty_values[key].append(obj_name)
        
        if len(missing_keys) == 0:
            objects_with_all_keys += 1
        else:
            objects_missing_keys.append((obj_name, list(missing_keys)))
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"   Total objects: {total_objects}")
    print(f"   Objects with ALL {len(EXPECTED_KEYS)} keys: {objects_with_all_keys} ({objects_with_all_keys/total_objects*100:.1f}%)")
    print(f"   Objects missing keys: {len(objects_missing_keys)} ({len(objects_missing_keys)/total_objects*100:.1f}%)")
    
    print(f"\n📋 KEY COVERAGE:")
    for key in EXPECTED_KEYS:
        coverage = key_coverage[key]
        percentage = (coverage / total_objects * 100) if total_objects > 0 else 0
        status = "✅" if percentage == 100 else "⚠️"
        print(f"   {status} {key:35s}: {coverage:4d}/{total_objects} ({percentage:5.1f}%)")
    
    print(f"\n🔍 EMPTY OR MISSING VALUES:")
    for key in EXPECTED_KEYS:
        empty_count = len(empty_values[key])
        if empty_count > 0:
            percentage = (empty_count / total_objects * 100) if total_objects > 0 else 0
            print(f"   ⚠️  {key:35s}: {empty_count:4d} objects ({percentage:5.1f}%)")
            if empty_count <= 5:
                print(f"       → {', '.join(empty_values[key])}")
            else:
                print(f"       → Sample: {', '.join(empty_values[key][:5])}...")
    
    if len(objects_missing_keys) > 0:
        print(f"\n❌ OBJECTS WITH MISSING KEYS (showing first 10):")
        for obj_name, missing in objects_missing_keys[:10]:
            if missing == "Not a dict":
                print(f"   • {obj_name}: {missing}")
            else:
                print(f"   • {obj_name}: missing {len(missing)} keys → {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}")
        
        if len(objects_missing_keys) > 10:
            print(f"   ... and {len(objects_missing_keys) - 10} more")
    
    print(f"\n{'='*80}\n")
    
    return {
        'total': total_objects,
        'complete': objects_with_all_keys,
        'incomplete': len(objects_missing_keys),
        'key_coverage': dict(key_coverage)
    }

def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "results" / "full_object_description"
    
    print("\n" + "="*80)
    print("VERIFYING DESCRIPTION COMPLETENESS")
    print("="*80)
    
    parsed_concepts_file = output_dir / "parsed_concepts_full_merged_complete.json"
    full_descriptions_file = output_dir / "full_object_descriptions_merged_complete.json"
    
    print("\n🔍 Checking if all objects have complete taxonomy structure...")
    
    stats1 = verify_description_completeness(parsed_concepts_file)
    stats2 = verify_description_completeness(full_descriptions_file)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nparsed_concepts_full_merged_complete.json:")
    print(f"   Complete objects: {stats1['complete']}/{stats1['total']} ({stats1['complete']/stats1['total']*100:.1f}%)")
    
    print(f"\nfull_object_descriptions_merged_complete.json:")
    print(f"   Complete objects: {stats2['complete']}/{stats2['total']} ({stats2['complete']/stats2['total']*100:.1f}%)")
    
    if stats1['complete'] == stats1['total'] and stats2['complete'] == stats2['total']:
        print("\n✅ ALL OBJECTS HAVE COMPLETE TAXONOMY STRUCTURE!")
    else:
        print("\n⚠️  SOME OBJECTS ARE MISSING TAXONOMY KEYS")

if __name__ == "__main__":
    main()

