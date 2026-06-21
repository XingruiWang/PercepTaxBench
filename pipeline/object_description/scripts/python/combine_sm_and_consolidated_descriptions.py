#!/usr/bin/env python3
"""
Combine SM descriptions (138 objects) with consolidated descriptions (800 objects).
Creates a unified object description file with both datasets.
"""

import json
import argparse
import os
from typing import Dict, Any

def combine_descriptions(sm_file: str, consolidated_file: str, output_file: str):
    """Combine SM and consolidated descriptions into a single file."""
    
    print(f"Loading SM descriptions from: {sm_file}")
    with open(sm_file, 'r') as f:
        sm_descriptions = json.load(f)
    
    print(f"Loading consolidated descriptions from: {consolidated_file}")
    with open(consolidated_file, 'r') as f:
        consolidated_descriptions = json.load(f)
    
    print(f"SM descriptions: {len(sm_descriptions)} objects")
    print(f"Consolidated descriptions: {len(consolidated_descriptions)} objects")
    
    # Check for overlaps
    sm_keys = set(sm_descriptions.keys())
    consolidated_keys = set(consolidated_descriptions.keys())
    overlap = sm_keys & consolidated_keys
    
    print(f"Overlapping objects: {len(overlap)}")
    if overlap:
        print(f"Overlapping objects: {list(overlap)[:10]}")
    
    # Combine descriptions
    combined_descriptions = {}
    
    # Add SM descriptions first
    for obj_name, obj_data in sm_descriptions.items():
        combined_descriptions[obj_name] = obj_data
    
    # Add consolidated descriptions (SM takes precedence for overlapping objects)
    for obj_name, obj_data in consolidated_descriptions.items():
        if obj_name not in combined_descriptions:
            combined_descriptions[obj_name] = obj_data
    
    print(f"Combined descriptions: {len(combined_descriptions)} objects")
    
    # Save combined descriptions
    with open(output_file, 'w') as f:
        json.dump(combined_descriptions, f, indent=2)
    
    print(f"Combined descriptions saved to: {output_file}")
    
    # Create summary
    summary = {
        "total_objects": len(combined_descriptions),
        "sm_objects": len(sm_descriptions),
        "consolidated_objects": len(consolidated_descriptions),
        "overlapping_objects": len(overlap),
        "unique_objects": len(combined_descriptions),
        "overlap_list": list(overlap) if overlap else []
    }
    
    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    print(f"Summary: {summary}")

def main():
    parser = argparse.ArgumentParser(description='Combine SM and consolidated descriptions')
    parser.add_argument('--sm_file', type=str, default='results/SM_descriptions.json',
                       help='Path to SM descriptions JSON file')
    parser.add_argument('--consolidated_file', type=str, default='results/consolidated_descriptions.json',
                       help='Path to consolidated descriptions JSON file')
    parser.add_argument('--output_file', type=str, default='results/combined_descriptions.json',
                       help='Output file for combined descriptions')
    
    args = parser.parse_args()
    
    combine_descriptions(args.sm_file, args.consolidated_file, args.output_file)

if __name__ == "__main__":
    main()
