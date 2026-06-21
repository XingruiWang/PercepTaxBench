#!/usr/bin/env python3
"""
Analyze detections from missing images to identify new objects
that need to be added to object descriptions and taxonomy.
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

def load_existing_object_descriptions(descriptions_path):
    """Load existing object descriptions"""
    with open(descriptions_path, 'r') as f:
        return json.load(f)

def analyze_missing_detections(missing_output_dir, existing_descriptions_path):
    """Analyze detections from missing images"""
    
    print("=" * 80)
    print("ANALYZING MISSING IMAGE DETECTIONS")
    print("=" * 80)
    print()
    
    # Load existing descriptions
    existing_objects = set()
    if os.path.exists(existing_descriptions_path):
        existing_descriptions = load_existing_object_descriptions(existing_descriptions_path)
        existing_objects = set(existing_descriptions.keys())
        print(f"Loaded {len(existing_objects)} existing object descriptions")
    else:
        print(f"WARNING: Existing descriptions file not found: {existing_descriptions_path}")
        existing_descriptions = {}
    
    print()
    
    # Analyze detections from missing images
    all_detected_objects = []
    object_counts = Counter()
    confidence_by_object = defaultdict(list)
    images_processed = 0
    images_with_detections = 0
    
    missing_output_path = Path(missing_output_dir)
    
    if not missing_output_path.exists():
        print(f"ERROR: Output directory does not exist: {missing_output_dir}")
        return
    
    # Iterate through all image directories
    for image_dir in missing_output_path.iterdir():
        if not image_dir.is_dir():
            continue
        
        images_processed += 1
        
        # Look for annotation file
        annotation_file = image_dir / "annotations" / f"{image_dir.name}.json"
        
        if not annotation_file.exists():
            continue
        
        # Load annotation
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)
        
        detections = annotation.get('detections', [])
        
        if detections:
            images_with_detections += 1
        
        for detection in detections:
            object_name = detection.get('object_name', '')
            class_name = detection.get('class_name', '')
            confidence = detection.get('confidence', 0.0)
            
            all_detected_objects.append({
                'object_name': object_name,
                'class_name': class_name,
                'confidence': confidence,
                'image': image_dir.name
            })
            
            object_counts[class_name] += 1
            confidence_by_object[class_name].append(confidence)
    
    print(f"Images processed: {images_processed}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {len(all_detected_objects)}")
    print()
    
    # Identify new objects (not in existing descriptions)
    detected_classes = set(object_counts.keys())
    new_objects = detected_classes - existing_objects
    existing_detected = detected_classes & existing_objects
    
    print("=" * 80)
    print("NEW OBJECTS (not in existing descriptions)")
    print("=" * 80)
    if new_objects:
        print(f"Found {len(new_objects)} new object classes:")
        print()
        for obj in sorted(new_objects):
            count = object_counts[obj]
            avg_conf = sum(confidence_by_object[obj]) / len(confidence_by_object[obj])
            print(f"  - {obj:30s} | Count: {count:3d} | Avg Confidence: {avg_conf:.3f}")
    else:
        print("No new objects found! All detected objects already have descriptions.")
    
    print()
    print("=" * 80)
    print("EXISTING OBJECTS (already have descriptions)")
    print("=" * 80)
    if existing_detected:
        print(f"Found {len(existing_detected)} existing object classes:")
        print()
        for obj in sorted(existing_detected):
            count = object_counts[obj]
            avg_conf = sum(confidence_by_object[obj]) / len(confidence_by_object[obj])
            print(f"  - {obj:30s} | Count: {count:3d} | Avg Confidence: {avg_conf:.3f}")
    else:
        print("No existing objects detected.")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique object classes: {len(detected_classes)}")
    print(f"New objects needing descriptions: {len(new_objects)}")
    print(f"Existing objects: {len(existing_detected)}")
    print()
    
    # Save results to JSON for further processing
    results = {
        'summary': {
            'images_processed': images_processed,
            'images_with_detections': images_with_detections,
            'total_detections': len(all_detected_objects),
            'unique_classes': len(detected_classes),
            'new_objects_count': len(new_objects),
            'existing_objects_count': len(existing_detected)
        },
        'new_objects': [
            {
                'class_name': obj,
                'count': object_counts[obj],
                'avg_confidence': sum(confidence_by_object[obj]) / len(confidence_by_object[obj]),
                'min_confidence': min(confidence_by_object[obj]),
                'max_confidence': max(confidence_by_object[obj])
            }
            for obj in sorted(new_objects)
        ],
        'existing_objects': [
            {
                'class_name': obj,
                'count': object_counts[obj],
                'avg_confidence': sum(confidence_by_object[obj]) / len(confidence_by_object[obj]),
                'min_confidence': min(confidence_by_object[obj]),
                'max_confidence': max(confidence_by_object[obj])
            }
            for obj in sorted(existing_detected)
        ],
        'all_detections': all_detected_objects
    }
    
    output_file = Path(missing_output_dir) / "detection_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed analysis saved to: {output_file}")
    print()
    
    # Create a list of new objects for easy processing
    if new_objects:
        new_objects_file = Path(missing_output_dir) / "new_objects_list.txt"
        with open(new_objects_file, 'w') as f:
            for obj in sorted(new_objects):
                f.write(f"{obj}\n")
        print(f"New objects list saved to: {new_objects_file}")
        print()
        print("NEXT STEPS:")
        print("1. Review the new objects list")
        print("2. Generate descriptions for new objects")
        print("3. Add new objects to taxonomy")
        print("4. Re-run QA generation with updated descriptions")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze detections from missing images")
    parser.add_argument("--missing_output_dir", type=str, 
                       default="/path/to/project/openimages_missing_output_low_conf",
                       help="Directory containing missing image detection results")
    parser.add_argument("--existing_descriptions", type=str,
                       default="/path/to/SpatialReasonerDataGen/object_description/results/filtered_full/parsed_concepts_filtered_full.json",
                       help="Path to existing object descriptions JSON")
    
    args = parser.parse_args()
    
    analyze_missing_detections(args.missing_output_dir, args.existing_descriptions)
