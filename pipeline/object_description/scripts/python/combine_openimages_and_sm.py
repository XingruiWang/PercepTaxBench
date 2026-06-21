#!/usr/bin/env python3
"""
Combine 800 consolidated OpenImages objects with 138 SM objects for unified processing.

This script creates a unified dataset that can be processed through the entire pipeline.
"""

import json
import os
import argparse
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_openimages_consolidated() -> List[str]:
    """Load the 800 consolidated OpenImages objects."""
    logger.info("Loading consolidated OpenImages objects")
    
    with open('object_description/results/consolidation_report.json', 'r') as f:
        data = json.load(f)
    
    consolidated_objects = data['consolidated_objects']
    logger.info(f"Loaded {len(consolidated_objects)} consolidated OpenImages objects")
    
    return consolidated_objects


def load_sm_objects() -> List[str]:
    """Load the 138 SM objects."""
    logger.info("Loading SM objects")
    
    with open('object_description/results/SM_descriptions.json', 'r') as f:
        sm_data = json.load(f)
    
    sm_objects = list(sm_data.keys())
    logger.info(f"Loaded {len(sm_objects)} SM objects")
    
    return sm_objects


def create_combined_dataset(oi_objects: List[str], sm_objects: List[str], output_file: str) -> Dict[str, Dict]:
    """Create combined dataset with both OpenImages and SM objects."""
    
    logger.info("Creating combined dataset")
    
    combined_descriptions = {}
    
    # Load existing OpenImages descriptions
    logger.info("Loading existing OpenImages descriptions")
    with open('object_description/results/object_descriptions_full.json', 'r') as f:
        oi_full_descriptions = json.load(f)
    
    # Load existing SM descriptions  
    logger.info("Loading existing SM descriptions")
    with open('object_description/results/SM_descriptions.json', 'r') as f:
        sm_descriptions = json.load(f)
    
    # Add OpenImages objects (using consolidated objects)
    consolidated_mapping = {}
    with open('object_description/results/consolidation_report.json', 'r') as f:
        oi_data = json.load(f)
    
    reverse_mapping = {}
    for original, consolidated in oi_data['consolidation_mapping'].items():
        if consolidated not in reverse_mapping:
            consolidated_mapping[consolidated] = []
        consolidated_mapping[consolidated].append(original)
    
    logger.info("Processing OpenImages objects")
    for obj in oi_objects:
        if obj in oi_full_descriptions:
            combined_descriptions[obj] = oi_full_descriptions[obj]
            logger.info(f"  ✓ Added OpenImages: {obj}")
        else:
            logger.warning(f"  ⚠ Missing description for OpenImages: {obj}")
    
    # Add SM objects
    logger.info("Processing SM objects") 
    for obj in sm_objects:
        if obj in sm_descriptions:
            combined_descriptions[obj] = sm_descriptions[obj]
            logger.info(f"  ✓ Added SM: {obj}")
        else:
            logger.warning(f"  ⚠ Missing description for SM: {obj}")
    
    # Save combined dataset
    logger.info(f"Saving combined dataset to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(combined_descriptions, f, indent=2)
    
    logger.info(f"Combined dataset created: {len(combined_descriptions)} objects")
    
    # Create summary report
    summary = {
        "total_objects": len(combined_descriptions),
        "openimages_objects": len([obj for obj in combined_descriptions.keys() if obj in oi_objects]),
        "sm_objects": len([obj for obj in combined_descriptions.keys() if obj in sm_objects]),
        "datasets": {
            "openimages_consolidated": len(oi_objects),
            "sm_objects": len(sm_objects),
            "combined_total": len(oi_objects) + len(sm_objects)
        },
        "descriptions_available": {
            "covers_all_objects": len(combined_descriptions) == len(oi_objects) + len(sm_objects),
            "missing_openimages": len(oi_objects) - len([obj for obj in oi_objects if obj in combined_descriptions]),
            "missing_sm": len(sm_objects) - len([obj for obj in sm_objects if obj in combined_descriptions])
        }
    }
    
    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved summary to {summary_file}")
    
    return combined_descriptions, summary


def create_object_list(combined_descriptions: Dict[str, Dict], output_file: str):
    """Create a simple list of all object names."""
    
    object_names = sorted(list(combined_descriptions.keys()))
    
    with open(output_file, 'w') as f:
        for name in object_names:
            f.write(f"{name}\n")
    
    logger.info(f"Created object list: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Combine OpenImages and SM datasets')
    parser.add_argument('--output_dir', 
                       default='object_description/results/combined',
                       help='Output directory for combined dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting dataset combination process")
    
    # Load datasets
    oi_objects = load_openimages_consolidated()
    sm_objects = load_sm_objects()
    
    print(f"\n=== DATASET SUMMARY ===")
    print(f"OpenImages consolidated: {len(oi_objects)} objects")
    print(f"SM objects: {len(sm_objects)} objects")
    print(f"Combined total: {len(oi_objects) + len(sm_objects)} objects")
    print()
    
    # Create combined dataset
    combined_file = os.path.join(args.output_dir, 'combined_object_descriptions.json')
    obj_list_file = os.path.join(args.output_dir, 'combined_objects.txt')
    
    combined_descriptions, summary = create_combined_dataset(oi_objects, sm_objects, combined_file)
    create_object_list(combined_descriptions, obj_list_file)
    
    # Create mapping files for reference
    oi_mapping = {obj: "openimages" for obj in oi_objects if obj in combined_descriptions}
    sm_mapping = {obj: "sm" for obj in sm_objects if obj in combined_descriptions}
    
    source_mapping_file = os.path.join(args.output_dir, 'source_mapping.json')
    source_mapping = {**oi_mapping, **sm_mapping}
    
    with open(source_mapping_file, 'w') as f:
        json.dump(source_mapping, f, indent=2)
    
    logger.info("Dataset combination completed!")
    
    print(f"\n=== RESULTS ===")
    print(f"Combined descriptions: {combined_file}")
    print(f"Object list: {obj_list_file}")
    print(f"Source mapping: {source_mapping_file}")
    print(f"Summary report: {combined_file.replace('.json', '_summary.json')}")
    print(f"Total objects: {len(combined_descriptions)}")
    print()
    
    # Show sample objects
    print("Sample combined objects:")
    sample_objects = sorted(list(combined_descriptions.keys()))[:10]
    for obj in sample_objects:
        source = source_mapping.get(obj, 'unknown')
        print(f"  {obj} ({source})")


if __name__ == "__main__":
    main()
