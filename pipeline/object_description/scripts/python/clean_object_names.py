#!/usr/bin/env python3
"""
Clean object names in the filtered combined object description and parsed JSON files:
1. Convert all object names to lowercase
2. Remove duplicates (keeping the first occurrence)
"""

import json
import logging
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_object_names(data: Dict) -> Dict:
    """Clean object names by converting to lowercase and removing duplicates."""
    cleaned_data = {}
    seen_names = set()
    duplicates_removed = []
    
    for obj_name, obj_data in data.items():
        # Convert to lowercase
        clean_name = obj_name.lower().strip()
        
        # Check for duplicates
        if clean_name in seen_names:
            duplicates_removed.append(f"{obj_name} -> {clean_name} (duplicate)")
            logger.warning(f"Removing duplicate: {obj_name} -> {clean_name}")
            continue
        
        # Add to cleaned data
        cleaned_data[clean_name] = obj_data
        seen_names.add(clean_name)
        
        # Log name changes
        if obj_name != clean_name:
            logger.info(f"Renamed: {obj_name} -> {clean_name}")
    
    if duplicates_removed:
        logger.info(f"Removed {len(duplicates_removed)} duplicates:")
        for dup in duplicates_removed:
            logger.info(f"  {dup}")
    
    return cleaned_data

def main():
    # Files to clean
    files_to_clean = [
        'object_description/results/filtered_full/object_descriptions_filtered_full.json',
        'object_description/results/filtered_full/parsed_concepts_filtered_full.json',
        'object_description/results/filtered_full/object_descriptions_filtered_full_source_mapping.json'
    ]
    
    for file_path in files_to_clean:
        logger.info(f"Cleaning {file_path}...")
        
        # Load the data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Original count: {len(data)} objects")
        
        # Clean the object names
        cleaned_data = clean_object_names(data)
        
        logger.info(f"Cleaned count: {len(cleaned_data)} objects")
        logger.info(f"Removed: {len(data) - len(cleaned_data)} objects")
        
        # Save the cleaned data
        with open(file_path, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        logger.info(f"Saved cleaned data to {file_path}")
        logger.info("")
    
    logger.info("Object name cleaning completed!")

if __name__ == "__main__":
    main()
