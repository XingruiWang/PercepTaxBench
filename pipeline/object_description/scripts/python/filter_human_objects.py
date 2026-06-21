#!/usr/bin/env python3
"""
Filter out human-related objects from OpenImages object statistics
"""

import json
import os

def filter_human_objects():
    # Define human-related objects to remove
    human_objects = {
        # Direct human references
        'person', 'man', 'woman', 'child', 'boy', 'girl', 'baby', 'toddler',
        'student', 'soldier', 'officer', 'player', 'singer', 'referee', 'skier', 
        'bride', 'groom', 'adult', 'kid', 'infant', 'teen', 'teenager',
        'fireman', 'businessman', 'businesswoman', 'policeman', 'policewoman',
        'sportsman', 'sportswoman', 'worker', 'employee', 'staff',
        
        # Human body parts
        'hand', 'face', 'eye', 'hair', 'beard', 'mouth', 'nose', 'ear',
        'head', 'arm', 'leg', 'foot', 'finger', 'thumb', 'neck', 'shoulder',
        'chest', 'back', 'belly', 'waist',
        
        # Human clothing and accessories
        'shirt', 'dress', 'suit', 'hat', 'helmet', 'goggles', 'sunglasses', 
        'glasses', 'shoe', 'sweatshirt', 'uniform', 'dress shirt', 't shirt',
        'cowboy hat', 'baseball hat', 'sun hat', 'bicycle helmet', 'safety vest',
        'swimwear', 'bikini', 'apron', 'costume', 'tie', 'gown', 'jacket',
        'pants', 'jeans', 'shorts', 'skirt', 'blouse', 'sweater', 'hoodie',
        'vest', 'coat', 'clothing', 'wear', 'attire', 'outfit', 'garment',
        'business suit', 'formal wear', 'casual wear', 'sportswear',
        'underwear', 'socks', 'gloves', 'mittens', 'scarf', 'belt',
        'necklace', 'bracelet', 'ring', 'earring', 'watch', 'jewelry',
        
        # Human groups/activities
        'crowd', 'audience', 'team', 'group', 'family', 'couple', 'pair',
        'band', 'choir', 'army', 'crew', 'staff', 'cast', 'ensemble',
        'rock band', 'football team', 'baseball team', 'basketball team',
        'party', 'wedding', 'graduation', 'meeting', 'conference',
        
        # Human-specific activities/roles
        'graduation', 'wedding', 'party', 'meeting', 'interview',
        'presentation', 'speech', 'performance', 'concert', 'show',
        'dance', 'dancing', 'singing', 'playing', 'running', 'walking',
        'sitting', 'standing', 'lying', 'sleeping', 'eating', 'drinking'
    }
    
    # Load original object stats
    input_file = 'openimages_3d_annotations/data/openimages_object_stats.json'
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    original_counts = data['object_counts']
    
    # Filter out human objects
    filtered_counts = {}
    removed_objects = []
    
    for obj, count in original_counts.items():
        obj_lower = obj.lower()
        
        # Check if object contains any human-related terms
        is_human_related = False
        for human_term in human_objects:
            if human_term.lower() in obj_lower or obj_lower in human_term.lower():
                is_human_related = True
                break
        
        if is_human_related:
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
    output_file = 'openimages_3d_annotations/data/openimages_object_stats_no_humans.json'
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    # Print summary
    print(f"Original objects: {data['total_unique_objects']}")
    print(f"Filtered objects: {len(filtered_counts)}")
    print(f"Removed objects: {len(removed_objects)}")
    print(f"Reduction: {filtered_data['filtering_info']['reduction_percentage']:.1f}%")
    print(f"\nRemoved human-related objects:")
    for obj in sorted(removed_objects)[:20]:  # Show first 20
        print(f"  - {obj} ({original_counts[obj]} detections)")
    if len(removed_objects) > 20:
        print(f"  ... and {len(removed_objects) - 20} more")
    
    print(f"\nFiltered object list saved to: {output_file}")
    
    # Create simplified object name list
    simple_list_file = 'openimages_3d_annotations/data/openimages_objects_no_humans_list.txt'
    with open(simple_list_file, 'w') as f:
        for obj in sorted(filtered_counts.keys()):
            f.write(f"{obj}\n")
    
    print(f"Simple object list saved to: {simple_list_file}")

if __name__ == "__main__":
    filter_human_objects()
