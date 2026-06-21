#!/usr/bin/env python3
"""
Verify the quality and completeness of OpenImages processing outputs.
This script checks for missing files, corrupted data, and incomplete processing.
"""

import os
import json
import sys
from typing import List, Dict, Any

def check_output_quality(output_dir: str, sample_size: int = 100) -> Dict[str, Any]:
    """Check the quality of processed outputs"""
    print(f"Checking output quality in: {output_dir}")
    
    # Get all processed directories
    processed_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    total_processed = len(processed_dirs)
    
    print(f"Total processed directories: {total_processed}")
    
    # Sample some directories for detailed checking
    import random
    sample_dirs = random.sample(processed_dirs, min(sample_size, total_processed))
    
    quality_report = {
        'total_processed': total_processed,
        'sample_checked': len(sample_dirs),
        'issues_found': [],
        'quality_stats': {
            'complete_outputs': 0,
            'missing_annotations': 0,
            'missing_crops': 0,
            'missing_visualizations': 0,
            'corrupted_json': 0,
            'empty_detections': 0
        }
    }
    
    for dir_name in sample_dirs:
        dir_path = os.path.join(output_dir, dir_name)
        issues = []
        
        # Check for required subdirectories
        required_dirs = ['annotations', 'object_crops', 'visualizations']
        for req_dir in required_dirs:
            req_path = os.path.join(dir_path, req_dir)
            if not os.path.exists(req_path):
                issues.append(f"Missing {req_dir} directory")
                quality_report['quality_stats'][f'missing_{req_dir}'] += 1
        
        # Check annotation files
        annotations_dir = os.path.join(dir_path, 'annotations')
        if os.path.exists(annotations_dir):
            json_file = os.path.join(annotations_dir, f'{dir_name}.json')
            summary_file = os.path.join(annotations_dir, f'{dir_name}_summary.txt')
            
            if not os.path.exists(json_file):
                issues.append("Missing JSON annotation file")
            elif not os.path.exists(summary_file):
                issues.append("Missing summary file")
            else:
                # Check JSON content
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    if 'detections' not in data:
                        issues.append("JSON missing detections field")
                    elif len(data['detections']) == 0:
                        issues.append("Empty detections")
                        quality_report['quality_stats']['empty_detections'] += 1
                    
                except json.JSONDecodeError:
                    issues.append("Corrupted JSON file")
                    quality_report['quality_stats']['corrupted_json'] += 1
                except Exception as e:
                    issues.append(f"Error reading JSON: {e}")
        
        # Check object crops (they may be in subdirectories)
        crops_dir = os.path.join(dir_path, 'object_crops')
        if os.path.exists(crops_dir):
            # Look for crop files recursively
            crop_files = []
            for root, dirs, files in os.walk(crops_dir):
                crop_files.extend([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
            if len(crop_files) == 0:
                issues.append("No crop files found")
        
        # Check visualizations
        viz_dir = os.path.join(dir_path, 'visualizations')
        if os.path.exists(viz_dir):
            viz_files = [f for f in os.listdir(viz_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if len(viz_files) == 0:
                issues.append("No visualization files found")
        
        if not issues:
            quality_report['quality_stats']['complete_outputs'] += 1
        else:
            quality_report['issues_found'].extend([(dir_name, issue) for issue in issues])
    
    return quality_report


def find_missing_images(source_dir: str, output_dir: str) -> List[str]:
    """Find images that haven't been processed yet"""
    print(f"Finding missing images...")
    
    # Get all source images
    source_images = []
    for f in os.listdir(source_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            source_images.append(os.path.splitext(f)[0])
    
    # Get processed images
    processed_images = []
    if os.path.exists(output_dir):
        processed_images = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    # Find missing
    missing_images = list(set(source_images) - set(processed_images))
    missing_images.sort()
    
    print(f"Source images: {len(source_images)}")
    print(f"Processed images: {len(processed_images)}")
    print(f"Missing images: {len(missing_images)}")
    
    return missing_images


def main():
    """Main verification function"""
    source_dir = "/path/to/project/openimages_train_10000"
    output_dir = "/path/to/project/openimages_unified_output"
    
    print("=== OpenImages Processing Verification ===")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if directories exist
    if not os.path.exists(source_dir):
        print(f"ERROR: Source directory does not exist: {source_dir}")
        return 1
    
    if not os.path.exists(output_dir):
        print(f"ERROR: Output directory does not exist: {output_dir}")
        return 1
    
    # Find missing images
    missing_images = find_missing_images(source_dir, output_dir)
    
    if missing_images:
        print(f"\nFirst 10 missing images:")
        for img in missing_images[:10]:
            print(f"  {img}")
        
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    
    print()
    
    # Check output quality
    quality_report = check_output_quality(output_dir)
    
    print("=== Quality Report ===")
    print(f"Total processed: {quality_report['total_processed']}")
    print(f"Sample checked: {quality_report['sample_checked']}")
    print()
    
    print("Quality Statistics:")
    for stat, count in quality_report['quality_stats'].items():
        percentage = (count / quality_report['sample_checked'] * 100) if quality_report['sample_checked'] > 0 else 0
        print(f"  {stat}: {count} ({percentage:.1f}%)")
    
    if quality_report['issues_found']:
        print(f"\nIssues found in {len(quality_report['issues_found'])} directories:")
        for dir_name, issue in quality_report['issues_found'][:10]:  # Show first 10 issues
            print(f"  {dir_name}: {issue}")
        
        if len(quality_report['issues_found']) > 10:
            print(f"  ... and {len(quality_report['issues_found']) - 10} more issues")
    else:
        print("\n✅ No quality issues found in sampled outputs!")
    
    print()
    print("=== Summary ===")
    completion_rate = ((quality_report['total_processed'] / (quality_report['total_processed'] + len(missing_images))) * 100) if (quality_report['total_processed'] + len(missing_images)) > 0 else 0
    print(f"Completion rate: {completion_rate:.1f}%")
    print(f"Missing images: {len(missing_images)}")
    print(f"Quality issues: {len(quality_report['issues_found'])}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
