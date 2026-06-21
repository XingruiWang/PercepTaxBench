#!/usr/bin/env python3
"""
Filter out human occupation types and related terms from the object list.
"""

import argparse
from pathlib import Path

def filter_occupations(input_file: str, output_file: str):
    """Filter out occupation-related objects from the input file."""
    
    # Define human occupation types and related terms
    occupation_terms = {
        'actor', 'athlete', 'barber', 'barista', 'bartender', 'baseball player', 'biker', 
        'bodybuilder', 'bridesmaid', 'businessman', 'carpenter', 'catcher', 'chef', 'choir',
        'client', 'coach', 'commuter', 'competitor', 'conductor', 'construction worker', 
        'cook', 'cowboy', 'crew', 'customer', 'dancer', 'daughter', 'decorate', 'design',
        'dj', 'doctor', 'drummer', 'engineer', 'fairy', 'family', 'father', 'fisherman',
        'fireman', 'geisha', 'graduate', 'grandmother', 'gymnast', 'hiker', 'hip hop artist',
        'jockey', 'journalist', 'judge', 'leader', 'magician', 'mechanic', 'mother', 
        'motorcyclist', 'mountain biker', 'nun', 'passenger', 'pedestrian', 'pilot', 'pirate',
        'pop artist', 'preacher', 'princess', 'professor', 'protester', 'rapper', 'rider',
        'rock artist', 'rock band', 'rock climber', 'runner', 'sailor', 'samurai', 'santa claus',
        'sculptor', 'skateboarder', 'skater', 'spectator', 'street artist', 'superhero',
        'surfer', 'teacher', 'tennis player', 'trainer', 'transporter', 'vendor', 'witch',
        'worker', 'wrestle', 'wrestler'
    }
    
    # Read input file
    with open(input_file, 'r') as f:
        objects = [line.strip() for line in f.readlines() if line.strip()]
    
    # Filter out occupation terms
    filtered_objects = [obj for obj in objects if obj not in occupation_terms]
    removed_objects = [obj for obj in objects if obj in occupation_terms]
    
    # Write filtered objects to output file
    with open(output_file, 'w') as f:
        for obj in filtered_objects:
            f.write(f"{obj}\n")
    
    # Print statistics
    print(f"=== OCCUPATION FILTERING RESULTS ===")
    print(f"Original objects: {len(objects)}")
    print(f"Removed occupation objects: {len(removed_objects)}")
    print(f"Remaining objects: {len(filtered_objects)}")
    print(f"Reduction: {len(removed_objects)} objects ({len(removed_objects)/len(objects)*100:.1f}%)")
    print()
    print(f"=== REMOVED OCCUPATION OBJECTS ===")
    for obj in sorted(removed_objects):
        print(f"  {obj}")
    print()
    print(f"Filtered objects saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Filter out human occupation types from object list")
    parser.add_argument("--input_file", required=True, help="Input object list file")
    parser.add_argument("--output_file", required=True, help="Output filtered object list file")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    filter_occupations(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
