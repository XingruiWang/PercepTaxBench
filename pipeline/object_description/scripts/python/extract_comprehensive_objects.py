#!/usr/bin/env python3
"""
Enhanced OpenImages Object Extraction Script

This script extracts all unique object names from OpenImages 3D annotations
and creates comprehensive metadata files for object description generation.
It ensures all newly detected objects are properly tracked and added to the
object descriptions pipeline.
"""

import json
import os
import glob
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

def extract_comprehensive_object_data(annotations_dir):
    """Extract comprehensive object data from OpenImages annotations."""
    object_names = set()
    object_counts = Counter()
    object_metadata = defaultdict(list)
    image_counts = defaultdict(int)
    
    # Find all annotation files
    annotation_files = glob.glob(os.path.join(annotations_dir, "*/annotations/*.json"))
    
    print(f"Found {len(annotation_files)} annotation files")
    
    for i, ann_file in enumerate(annotation_files):
        if i % 1000 == 0:
            print(f"Processing file {i+1}/{len(annotation_files)}")
            
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
                
            image_name = data.get('image_name', 'unknown')
            
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
                        
                        # Store metadata
                        detection_metadata = {
                            'image_name': image_name,
                            'confidence': detection.get('confidence', 0.0),
                            'bbox': detection.get('bbox', []),
                            'detection_id': detection.get('detection_id', ''),
                            'original_name': obj_name
                        }
                        object_metadata[clean_name].append(detection_metadata)
                        
                        # Count unique images per object
                        if image_name not in [meta['image_name'] for meta in object_metadata[clean_name][:-1]]:
                            image_counts[clean_name] += 1
                        
        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
            continue
    
    return object_names, object_counts, object_metadata, image_counts

def create_comprehensive_object_database(object_names, object_counts, object_metadata, image_counts, output_dir):
    """Create comprehensive object database with metadata."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Simple text list (for backward compatibility)
    txt_file = os.path.join(output_dir, "openimages_detected_objects.txt")
    with open(txt_file, 'w') as f:
        for name in sorted(object_names):
            f.write(f"{name}\n")
    
    # 2. Comprehensive JSON database
    json_file = os.path.join(output_dir, "openimages_detected_objects_database.json")
    
    database = {
        "metadata": {
            "extraction_date": datetime.now().isoformat(),
            "total_unique_objects": len(object_names),
            "total_detections": sum(object_counts.values()),
            "total_images_processed": len(set(meta['image_name'] for obj_metas in object_metadata.values() for meta in obj_metas)),
            "source_directory": "openimages_unified_output"
        },
        "objects": {}
    }
    
    # Add each object with comprehensive metadata
    for obj_name in sorted(object_names):
        detections = object_metadata[obj_name]
        confidences = [d['confidence'] for d in detections]
        
        database["objects"][obj_name] = {
            "total_detections": object_counts[obj_name],
            "unique_images": image_counts[obj_name],
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "min_confidence": min(confidences) if confidences else 0.0,
            "max_confidence": max(confidences) if confidences else 0.0,
            "detection_details": detections[:10]  # Keep first 10 detections for reference
        }
    
    with open(json_file, 'w') as f:
        json.dump(database, f, indent=2)
    
    # 3. Statistics summary
    stats_file = os.path.join(output_dir, "openimages_object_stats.json")
    stats = {
        "total_unique_objects": len(object_names),
        "total_detections": sum(object_counts.values()),
        "total_images": len(set(meta['image_name'] for obj_metas in object_metadata.values() for meta in obj_metas)),
        "object_counts": dict(object_counts.most_common()),
        "most_common_objects": dict(object_counts.most_common(50)),
        "objects_by_image_count": dict(sorted(image_counts.items(), key=lambda x: x[1], reverse=True)[:50]),
        "extraction_date": datetime.now().isoformat()
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return txt_file, json_file, stats_file

def main():
    """Main function to extract and organize OpenImages object data."""
    
    # Paths
    annotations_dir = "../../openimages_unified_output"
    output_dir = "openimages_3d_annotations/data"
    
    print("=" * 60)
    print("Enhanced OpenImages Object Extraction")
    print("=" * 60)
    print(f"Source directory: {annotations_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Extract comprehensive object data
    print("Extracting object data from annotations...")
    object_names, object_counts, object_metadata, image_counts = extract_comprehensive_object_data(annotations_dir)
    
    print(f"\nExtraction Results:")
    print(f"  Unique objects: {len(object_names)}")
    print(f"  Total detections: {sum(object_counts.values())}")
    print(f"  Total images: {len(set(meta['image_name'] for obj_metas in object_metadata.values() for meta in obj_metas))}")
    
    # Create comprehensive database
    print(f"\nCreating comprehensive object database...")
    txt_file, json_file, stats_file = create_comprehensive_object_database(
        object_names, object_counts, object_metadata, image_counts, output_dir
    )
    
    print(f"\nFiles created:")
    print(f"  Text list: {txt_file}")
    print(f"  JSON database: {json_file}")
    print(f"  Statistics: {stats_file}")
    
    print(f"\nTop 20 most detected objects:")
    for obj, count in object_counts.most_common(20):
        unique_images = image_counts[obj]
        print(f"  {obj}: {count} detections in {unique_images} images")
    
    print(f"\nTop 20 objects by unique image count:")
    for obj, img_count in sorted(image_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        detections = object_counts[obj]
        print(f"  {obj}: {img_count} images ({detections} total detections)")
    
    print(f"\n✓ Object extraction completed successfully!")
    print(f"✓ All files saved to: {output_dir}")
    print(f"✓ Ready for object description generation!")

if __name__ == "__main__":
    main()
