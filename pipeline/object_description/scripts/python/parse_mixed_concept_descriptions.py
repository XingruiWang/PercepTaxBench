#!/usr/bin/env python3
"""
Parse Mixed Concept Descriptions Script

This script takes object description JSON files and uses Gemini Flash 2.5 as a judge
to split mixed concept descriptions into separate concept-specific descriptions.
The output maintains the same JSON format but with parsed descriptions.

Usage:
    python parse_mixed_concept_descriptions.py --input_file <path> --output_dir <path> --api_key <key>
"""

import json
import argparse
import os
import logging
from typing import Dict, List, Any
import google.generativeai as genai
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MixedConceptParser:
    """Parser for mixed concept descriptions using Gemini Flash 2.5 as judge."""
    
    def __init__(self, api_key: str):
        """Initialize the parser with Gemini API key."""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Define the semantic keys we're working with
        self.semantic_keys = [
            'general_description',
            'shape', 
            'texture',
            'color',
            'material',
            'functions',
            'affordance',
            'common_elements',
            'common_environmental_context',
            'additional_details',
            'physical_properties'
        ]
        
    def create_parsing_prompt(self, object_name: str, key: str, description: str) -> str:
        """Create a prompt for Gemini to parse mixed concept descriptions."""
        
        prompt = f"""You are an expert at analyzing object descriptions and identifying mixed concepts.

OBJECT: {object_name}
SEMANTIC KEY: {key}
ORIGINAL DESCRIPTION: "{description}"

TASK: Analyze the description above and determine if it contains mixed concepts (multiple distinct concepts within the same description). If it does, split it into separate concept-specific descriptions.

RULES:
1. If the description contains only ONE concept, return it unchanged
2. If the description contains MULTIPLE concepts, split them into separate descriptions
3. Each concept should be clearly separated and focused on the semantic key
4. Maintain the original meaning and detail level
5. Use clear, concise language for each concept

OUTPUT FORMAT:
- If single concept: Return the original description
- If multiple concepts: Return each concept as a separate bullet point:
  • Concept 1 description
  • Concept 2 description
  • Concept 3 description

EXAMPLES:

INPUT: "Smooth metal surface with rough wooden handles"
OUTPUT: 
• Smooth metal surface
• Rough wooden handles

INPUT: "Red and blue colored fabric"
OUTPUT: Red and blue colored fabric

INPUT: "Used for sitting, storage, and decoration"
OUTPUT:
• Used for sitting
• Used for storage  
• Used for decoration

INPUT: "Heavy, rigid, stable, solid"
OUTPUT:
• Heavy
• Rigid
• Stable
• Solid

Description sentence with "\"  should be directlysplit into separate concepts.
Descriptions that are words used to describe a concept instead of a sentence should be split into separate concepts.

Now analyze the description for "{object_name}" under "{key}":"""

        return prompt
    
    def parse_description(self, object_name: str, key: str, description: str) -> List[str]:
        """Parse a single description using Gemini."""
        
        try:
            prompt = self.create_parsing_prompt(object_name, key, description)
            
            response = self.model.generate_content(prompt)
            parsed_text = response.text.strip()
            
            # Split by bullet points if multiple concepts
            if '•' in parsed_text:
                concepts = [concept.strip() for concept in parsed_text.split('•') if concept.strip()]
                logger.info(f"Parsed {object_name} {key}: {len(concepts)} concepts found")
                return concepts
            else:
                # Single concept, return as list with one item
                logger.info(f"Parsed {object_name} {key}: single concept")
                return [parsed_text]
                
        except Exception as e:
            logger.error(f"Error parsing {object_name} {key}: {str(e)}")
            # Return original description if parsing fails
            return [description]
    
    def parse_object_descriptions(self, object_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse all descriptions for a single object."""
        
        parsed_object = {}
        
        for key in self.semantic_keys:
            if key in object_data:
                original_description = object_data[key]
                
                # Parse the description
                parsed_concepts = self.parse_description(
                    list(object_data.keys())[0] if len(object_data) == 1 else "Unknown",
                    key, 
                    original_description
                )
                
                # Store parsed concepts
                parsed_object[key] = parsed_concepts
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
            else:
                parsed_object[key] = [object_data.get(key, "")]
        
        return parsed_object
    
    def parse_file(self, input_file: str, output_file: str, resume: bool = True) -> None:
        """Parse an entire description file with incremental saving and resume capability."""
        
        logger.info(f"Loading input file: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} objects")
        
        # Check for existing progress file
        progress_file = output_file.replace('.json', '_progress.json')
        parsed_data = {}
        start_index = 0
        
        if resume and os.path.exists(progress_file):
            logger.info(f"Found existing progress file: {progress_file}")
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    parsed_data = json.load(f)
                start_index = len(parsed_data)
                logger.info(f"Resuming from object {start_index + 1}/{len(data)}")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}. Starting from beginning.")
                parsed_data = {}
                start_index = 0
        
        total_objects = len(data)
        processed_count = 0
        
        # Process objects starting from resume point
        for i, (object_name, object_data) in enumerate(data.items(), 1):
            if i <= start_index:
                continue
                
            logger.info(f"Processing {object_name} ({i}/{total_objects})")
            
            try:
                parsed_object = self.parse_object_descriptions(object_data)
                parsed_data[object_name] = parsed_object
                processed_count += 1
                
                # Save progress every 5 objects
                if processed_count % 5 == 0:
                    self._save_progress(parsed_data, progress_file)
                    logger.info(f"Progress saved: {len(parsed_data)}/{total_objects} objects processed")
                    
            except Exception as e:
                logger.error(f"Error processing {object_name}: {str(e)}")
                # Keep original data if parsing fails
                parsed_data[object_name] = object_data
                processed_count += 1
        
        # Save final parsed data
        logger.info(f"Saving final parsed data to: {output_file}")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, indent=4, ensure_ascii=False)
        
        # Clean up progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
            logger.info("Progress file cleaned up")
        
        logger.info(f"Successfully parsed and saved {len(parsed_data)} objects")
    
    def _save_progress(self, parsed_data: Dict[str, Any], progress_file: str) -> None:
        """Save current progress to a temporary file."""
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")

def main():
    """Main function to run the parsing script."""
    
    parser = argparse.ArgumentParser(description='Parse mixed concept descriptions using Gemini Flash 2.5')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input JSON file with object descriptions')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save parsed descriptions')
    parser.add_argument('--api_key', type=str, required=True,
                       help='Gemini API key')
    parser.add_argument('--output_filename', type=str, default=None,
                       help='Custom output filename (default: parsed_<input_filename>)')
    parser.add_argument('--no_resume', action='store_true',
                       help='Disable resume functionality and start from beginning')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output filename
    if args.output_filename:
        output_filename = args.output_filename
    else:
        input_basename = os.path.basename(args.input_file)
        name, ext = os.path.splitext(input_basename)
        output_filename = f"parsed_{name}{ext}"
    
    output_file = os.path.join(args.output_dir, output_filename)
    
    # Initialize parser and run
    parser_instance = MixedConceptParser(args.api_key)
    parser_instance.parse_file(args.input_file, output_file, resume=not args.no_resume)
    
    logger.info("Parsing completed successfully!")

if __name__ == "__main__":
    main()
