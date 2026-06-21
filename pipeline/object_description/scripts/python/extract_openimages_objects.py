#!/usr/bin/env python3
"""
Extract all unique object names from OpenImages annotations
and update the object_descriptions_full output for further processing.
"""

import json
import os
import glob
from collections import Counter
from pathlib import Path

def extract_object_names_from_annotations(annotations_dir):
    """Extract all unique object names from OpenImages annotations."""
    object_names = set()
    object_counts = Counter()
    
    # Find all annotation files
    annotation_files = glob.glob(os.path.join(annotations_dir, "*/annotations/*.json"))
    
    print(f"Found {len(annotation_files)} annotation files")
    
    for i, ann_file in enumerate(annotation_files):
        if i % 1000 == 0:
            print(f"Processing file {i+1}/{len(annotation_files)}")
            
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
                
            # Extract object names from detections
            if 'detections' in data:
                for detection in data['detections']:
                    if 'object_name' in detection:
                        # Extract the actual object name (remove obj_XX_ prefix)
                        obj_name = detection['object_name']
                        if obj_name.startswith('obj_'):
                            # Remove obj_XX_ prefix to get clean object name
                            clean_name = obj_name.split('_', 2)[-1] if '_' in obj_name else obj_name
                        else:
                            clean_name = obj_name
                        
                        object_names.add(clean_name)
                        object_counts[clean_name] += 1
                        
        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
            continue
    
    return object_names, object_counts

def create_object_descriptions_list(object_names, output_file):
    """Create a list of object names for description generation."""
    sorted_names = sorted(object_names)
    
    with open(output_file, 'w') as f:
        for name in sorted_names:
            f.write(f"{name}\n")
    
    print(f"Created object list with {len(sorted_names)} unique objects: {output_file}")

def main():
    # Paths
    annotations_dir = "openimages_unified_output"
    output_file = "openimages_detected_objects.txt"
    stats_file = "openimages_object_stats.json"
    
    print("Extracting object names from OpenImages annotations...")
    
    # Extract object names
    object_names, object_counts = extract_object_names_from_annotations(annotations_dir)
    
    print(f"\nFound {len(object_names)} unique object types")
    print(f"Total detections: {sum(object_counts.values())}")
    
    # Create object list file
    create_object_descriptions_list(object_names, output_file)
    
    # Save statistics
    stats = {
        "total_unique_objects": len(object_names),
        "total_detections": sum(object_counts.values()),
        "object_counts": dict(object_counts.most_common()),
        "most_common_objects": dict(object_counts.most_common(20))
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_file}")
    print(f"\nTop 20 most detected objects:")
    for obj, count in object_counts.most_common(20):
        print(f"  {obj}: {count}")
    
    print(f"\nObject list created: {output_file}")
    print(f"Ready for object description generation!")

if __name__ == "__main__":
    main()
