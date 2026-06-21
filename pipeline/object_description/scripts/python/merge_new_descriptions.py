#!/usr/bin/env python3
"""
Merge newly generated descriptions with existing descriptions
"""

import json
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results" / "full_object_description"
    
    # Load existing descriptions
    existing_path = results_dir / "full_object_descriptions_fully_parsed.json"
    with open(existing_path, 'r') as f:
        existing_descriptions = json.load(f)
    
    print(f"Loaded {len(existing_descriptions)} existing descriptions")
    
    # Load new descriptions
    new_path = results_dir / "new_object_descriptions.json"
    if not new_path.exists():
        print(f"Error: {new_path} not found")
        return
    
    with open(new_path, 'r') as f:
        new_descriptions = json.load(f)
    
    print(f"Loaded {len(new_descriptions)} new descriptions")
    
    # Merge (new descriptions override existing if there's overlap)
    merged = existing_descriptions.copy()
    overlap_count = 0
    
    for obj_name, description in new_descriptions.items():
        if obj_name in merged:
            overlap_count += 1
        merged[obj_name] = description
    
    print(f"\nMerge summary:")
    print(f"  Existing: {len(existing_descriptions)}")
    print(f"  New: {len(new_descriptions)}")
    print(f"  Overlap (updated): {overlap_count}")
    print(f"  Total after merge: {len(merged)}")
    
    # Backup existing file
    backup_path = results_dir / "full_object_descriptions_fully_parsed_backup.json"
    print(f"\nBacking up existing to: {backup_path.name}")
    with open(backup_path, 'w') as f:
        json.dump(existing_descriptions, f, indent=2)
    
    # Save merged file
    print(f"Saving merged descriptions to: {existing_path.name}")
    with open(existing_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\n✅ Merge complete!")
    print(f"   Coverage: {len(merged)} objects with descriptions")

if __name__ == "__main__":
    main()

