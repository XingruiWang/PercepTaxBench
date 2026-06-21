#!/usr/bin/env python3
"""
Filter out only direct human references and human body parts from OpenImages object statistics
"""

import json
import os

def filter_direct_humans():
    # Define only direct human references and body parts to remove
    direct_human_objects = {
        # Direct human references
        'person', 'man', 'woman', 'child', 'boy', 'girl', 'baby', 'toddler',
        'student', 'soldier', 'officer', 'player', 'singer', 'referee', 'skier', 
        'bride', 'groom', 'adult', 'kid', 'infant', 'teen', 'teenager',
        
        # Human body parts
        'hand', 'face', 'eye', 'hair', 'beard', 'mouth', 'nose', 'ear',
        'head', 'arm', 'leg', 'foot', 'finger', 'thumb', 'neck', 'shoulder',
        'chest', 'back', 'belly', 'waist', 'knee', 'elbow', 'wrist', 'ankle',
        'toe', 'heel', 'palm', 'fist', 'forehead', 'cheek', 'chin', 'lip',
        'tooth', 'tongue', 'eyebrow', 'eyelash', 'mustache', 'sideburn'
    }
    
    # Load original object stats
    input_file = 'openimages_3d_annotations/data/openimages_object_stats.json'
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    original_counts = data['object_counts']
    
    # Filter out direct human objects
    filtered_counts = {}
    removed_objects = []
    
    for obj, count in original_counts.items():
        obj_lower = obj.lower().strip()
        
        # Check if object is exactly in the direct human list
        if obj_lower in direct_human_objects:
            removed_objects.append(obj)
        else:
            filtered_counts[obj] = count
    
    # Create filtered data
    filtered_data = {
        'total_unique_objects': len(filtered_counts),
        'total_detections': sum(filtered_counts.values()),
        'total_images': data['total_images'],
        'object_counts': filtered_counts,
        'filtering_info': {
            'original_objects': data['total_unique_objects'],
            'filtered_objects': len(filtered_counts),
            'removed_objects': len(removed_objects),
            'removed_object_list': sorted(removed_objects),
            'reduction_percentage': (len(removed_objects) / data['total_unique_objects']) * 100
        }
    }
    
    # Save filtered data
    output_file = 'openimages_3d_annotations/data/openimages_object_stats_filtered.json'
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    # Print summary
    print(f"Original objects: {data['total_unique_objects']}")
    print(f"Filtered objects: {len(filtered_counts)}")
    print(f"Removed objects: {len(removed_objects)}")
    print(f"Reduction: {filtered_data['filtering_info']['reduction_percentage']:.1f}%")
    print(f"\nRemoved direct human objects:")
    for obj in sorted(removed_objects):
        print(f"  - {obj} ({original_counts[obj]} detections)")
    
    print(f"\nFiltered object list saved to: {output_file}")
    
    # Create simplified object name list
    simple_list_file = 'openimages_3d_annotations/data/openimages_objects_filtered_list.txt'
    with open(simple_list_file, 'w') as f:
        for obj in sorted(filtered_counts.keys()):
            f.write(f"{obj}\n")
    
    print(f"Simple object list saved to: {simple_list_file}")

if __name__ == "__main__":
    filter_direct_humans()

