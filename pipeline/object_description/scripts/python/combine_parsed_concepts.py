#!/usr/bin/env python3
"""
Combine parsed concepts from OpenImages and SM datasets.

This script combines existing parsed concepts from both datasets 
into a unified dataset for HDBSCAN clustering.
"""

import json
import os
import argparse
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_openimages_parsed() -> Dict[str, Dict]:
    """Load parsed concepts from OpenImages dataset."""
    logger.info("Loading OpenImages parsed concepts")
    
    with open('object_description/results/consolidated_descriptions_parsed.json', 'r') as f:
        oi_parsed = json.load(f)
    
    logger.info(f"Loaded {len(oi_parsed)} OpenImages parsed concepts")
    return oi_parsed


def load_sm_parsed() -> Dict[str, Dict]:
    """Load parsed concepts from SM dataset."""
    logger.info("Loading SM parsed concepts")
    
    with open('object_description/results/parsed/parsed_SM_descriptions.json', 'r') as f:
        sm_parsed = json.load(f)
    
    logger.info(f"Loaded {len(sm_parsed)} SM parsed concepts")
    return sm_parsed


def combine_parsed_concepts(oi_parsed: Dict[str, Dict], 
                          sm_parsed: Dict[str, Dict],
                          source_mapping: Dict[str, str],
                          output_file: str) -> Dict[str, Dict]:
    """Combine parsed concepts from both datasets."""
    
    logger.info("Combining parsed concepts")
    
    combined_parsed = {}
    
    # Add OpenImages parsed concepts
    logger.info("Processing OpenImages parsed concepts")
    for obj_name, parsed_data in oi_parsed.items():
        if source_mapping.get(obj_name) == "openimages":
            combined_parsed[obj_name] = parsed_data
            logger.debug(f"  ✓ Added OpenImages parsed: {obj_name}")
        else:
            logger.warning(f"  ⚠ Object not in source mapping: {obj_name}")
    
    # Add SM parsed concepts
    logger.info("Processing SM parsed concepts")
    for obj_name, parsed_data in sm_parsed.items():
        if source_mapping.get(obj_name) == "sm":
            combined_parsed[obj_name] = parsed_data
            logger.debug(f"  ✓ Added SM parsed: {obj_name}")
        else:
            logger.warning(f"  ⚠ Object not in source mapping: {obj_name}")
    
    # Save combined parsed concepts
    logger.info(f"Saving combined parsed concepts to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(combined_parsed, f, indent=2)
    
    logger.info(f"Combined parsed concepts: {len(combined_parsed)} objects")
    
    return combined_parsed


def create_full_object_descriptions(combined_descriptions: Dict[str, Dict], 
                                  output_file: str):
    """Create full object descriptions file (renamed)."""
    
    logger.info(f"Creating full object descriptions: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(combined_descriptions, f, indent=2)
    
    logger.info(f"Created full object descriptions: {len(combined_descriptions)} objects")


def analyze_parsed_concepts(combined_parsed: Dict[str, Dict]) -> Dict:
    """Analyze the combined parsed concepts structure."""
    
    logger.info("Analyzing parsed concepts structure")
    
    analysis = {
        "total_objects": len(combined_parsed),
        "key_analysis": {},
        "sample_keys": [],
        "data_quality": {}
    }
    
    # Analyze available keys
    all_keys = set()
    objects_with_keys = {}
    
    for obj_name, parsed_data in combined_parsed.items():
        if isinstance(parsed_data, dict):
            for key in parsed_data.keys():
                all_keys.add(key)
                if key not in objects_with_keys:
                    objects_with_keys[key] = []
                objects_with_keys[key].append(obj_name)
    
    analysis["sample_keys"] = sorted(list(all_keys))
    
    # Key analysis
    for key in all_keys:
        objects_with_this_key = objects_with_keys[key]
        analysis["key_analysis"][key] = {
            "object_count": len(objects_with_this_key),
            "coverage": len(objects_with_this_key) / len(combined_parsed),
            "sample_objects": objects_with_this_key[:5]
        }
    
    # Data quality analysis
    complete_objects = 0
    for obj_name, parsed_data in combined_parsed.items():
        if isinstance(parsed_data, dict) and len(parsed_data) >= 5:
            complete_objects += 1
    
    analysis["data_quality"] = {
        "complete_objects": complete_objects,
        "completion_rate": complete_objects / len(combined_parsed),
        "total_keys": len(all_keys)
    }
    
    logger.info("Parsed concepts analysis completed")
    return analysis


def main():
    parser = argparse.ArgumentParser(description='Combine parsed concepts from OpenImages and SM datasets')
    parser.add_argument('--output_dir', 
                       default='object_description/results/combined',
                       help='Output directory for combined data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting parsed concepts combination process")
    
    # Load parsed concepts
    oi_parsed = load_openimages_parsed()
    sm_parsed = load_sm_parsed()
    
    # Load source mapping
    with open(os.path.join(args.output_dir, 'source_mapping.json'), 'r') as f:
        source_mapping = json.load(f)
    
    # Load combined descriptions
    with open(os.path.join(args.output_dir, 'combined_object_descriptions.json'), 'r') as f:
        combined_descriptions = json.load(f)
    
    print(f"\n=== PARSED CONCEPTS COMBINATION ===")
    print(f"OpenImages parsed: {len(oi_parsed)} objects")
    print(f"SM parsed: {len(sm_parsed)} objects")
    print(f"Combined descriptions: {len(combined_descriptions)} objects")
    print()
    
    # Combine parsed concepts
    parsed_file = os.path.join(args.output_dir, 'combined_parsed_concepts.json')
    combined_parsed = combine_parsed_concepts(oi_parsed, sm_parsed, source_mapping, parsed_file)
    
    # Create full object descriptions (renamed)
    full_descriptions_file = os.path.join(args.output_dir, 'full_object_descriptions.json')
    create_full_object_descriptions(combined_descriptions, full_descriptions_file)
    
    # Analyze parsed concepts
    analysis_file = os.path.join(args.output_dir, 'parsed_concepts_analysis.json')
    analysis = analyze_parsed_concepts(combined_parsed)
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info("Parsed concepts combination completed!")
    
    print(f"\n=== RESULTS ===")
    print(f"Combined parsed concepts: {parsed_file}")
    print(f"Full object descriptions: {full_descriptions_file}")
    print(f"Analysis report: {analysis_file}")
    print(f"Total parsed objects: {len(combined_parsed)}")
    print()
    
    # Show key analysis
    print("Available keys for clustering:")
    for key, data in analysis["key_analysis"].items():
        print(f"  {key}: {data['object_count']} objects ({data['coverage']:.1%} coverage)")
    
    print(f"\nData quality:")
    print(f"  Complete objects: {analysis['data_quality']['complete_objects']}")
    print(f"  Completion rate: {analysis['data_quality']['completion_rate']:.1%}")
    print(f"  Total keys: {analysis['data_quality']['total_keys']}")
    
    print(f"\n🎯 Ready for HDBSCAN clustering!")


if __name__ == "__main__":
    main()
