#!/usr/bin/env python3
"""
Analyze name mismatches in sim image generation:
1. Compare seenable_obj_dict.json vs object_annots.json for each scene
2. Check sm_to_human_mapping coverage
3. Identify naming pattern issues
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

sys.path.append('/path/to/SpatialReasonerDataGen/qa_gen/scripts')

from modules.qa_modules.data_loading_utils import DataLoadingUtils

def analyze_scene_mismatches(scene_path: Path, sm_to_human_mapping: Dict[str, str]) -> Dict:
    """Analyze mismatches for a single scene"""
    results = {
        'scene_id': scene_path.name,
        'seenable_obj_count': 0,
        'object_annots_count': 0,
        'seenable_in_annots': 0,
        'annots_in_seenable': 0,
        'missing_in_seenable': [],
        'missing_in_annots': [],
        'mapping_coverage': 0,
        'unmapped_seenable': [],
        'unmapped_annots': []
    }
    
    # Load seenable_obj_dict.json
    seenable_obj_file = scene_path / "seenable_obj_dict.json"
    seenable_obj_dict = {}
    if seenable_obj_file.exists():
        try:
            with open(seenable_obj_file, 'r') as f:
                seenable_obj_dict = json.load(f)
            results['seenable_obj_count'] = len(seenable_obj_dict)
        except Exception as e:
            print(f"ERROR loading seenable_obj_dict for {scene_path.name}: {e}")
            return results
    
    # Load object_annots.json
    object_annots_file = scene_path / "object_annots.json"
    annots_obj_ids = set()
    if object_annots_file.exists():
        try:
            with open(object_annots_file, 'r', encoding='utf-8') as f:
                annots_data = json.load(f)
            
            for obj_data in annots_data.get('outputs', []):
                obj_id = obj_data.get('object_id', '')
                if obj_id:
                    # Filter out lighting/debug objects
                    if not any(x in obj_id.lower() for x in ['light', 'debug', 'capture', 'fog', 'sky', 'game', 'player', 'world', 'landscape', 'exponential', 'atmospheric', 'chaos', 'gameplay']):
                        annots_obj_ids.add(obj_id)
            
            results['object_annots_count'] = len(annots_obj_ids)
        except Exception as e:
            print(f"ERROR loading object_annots for {scene_path.name}: {e}")
            return results
    
    # Compare sets
    seenable_obj_ids = set(seenable_obj_dict.keys())
    
    # Objects in seenable but not in annots
    results['missing_in_annots'] = list(seenable_obj_ids - annots_obj_ids)
    results['seenable_in_annots'] = len(seenable_obj_ids & annots_obj_ids)
    
    # Objects in annots but not in seenable
    results['missing_in_seenable'] = list(annots_obj_ids - seenable_obj_ids)
    results['annots_in_seenable'] = len(annots_obj_ids & seenable_obj_ids)
    
    # Check mapping coverage with fuzzy matching
    mapped_seenable = 0
    mapped_annots = 0
    
    for obj_id in seenable_obj_ids:
        # Try exact match
        if obj_id in sm_to_human_mapping:
            mapped_seenable += 1
        else:
            # Try fuzzy match
            base_name = re.sub(r'_?\d+(_\d+)*$', '', obj_id)
            if base_name in sm_to_human_mapping:
                mapped_seenable += 1
            else:
                results['unmapped_seenable'].append(obj_id)
    
    for obj_id in annots_obj_ids:
        # Try exact match
        if obj_id in sm_to_human_mapping:
            mapped_annots += 1
        else:
            # Try fuzzy match
            base_name = re.sub(r'_?\d+(_\d+)*$', '', obj_id)
            if base_name in sm_to_human_mapping:
                mapped_annots += 1
            else:
                results['unmapped_annots'].append(obj_id)
    
    if seenable_obj_ids:
        results['mapping_coverage'] = mapped_seenable / len(seenable_obj_ids) * 100
    
    return results

def analyze_all_scenes(images_dir: Path, max_scenes: int = None) -> Dict:
    """Analyze all scenes in the images directory"""
    data_loader = DataLoadingUtils()
    sm_to_human_mapping = data_loader.load_sm_to_human_mapping()
    
    print("=" * 80)
    print("SIM IMAGE NAME MISMATCH ANALYSIS")
    print("=" * 80)
    print(f"\nLoaded {len(sm_to_human_mapping)} SM to human mappings")
    print(f"Scanning scenes in: {images_dir}\n")
    
    # Find all scene directories (handle nested structure: scene_collection/scene_id/)
    scene_paths = []
    for collection_dir in images_dir.iterdir():
        if collection_dir.is_dir() and not collection_dir.name.startswith('.') and not collection_dir.name.endswith('.zip'):
            # Check if this is a scene collection or individual scene
            for scene_dir in collection_dir.iterdir():
                if scene_dir.is_dir() and not scene_dir.name.startswith('.'):
                    # Check if it has required files
                    if (scene_dir / "seenable_obj_dict.json").exists() and (scene_dir / "object_annots.json").exists():
                        scene_paths.append(scene_dir)
                    # Also check if collection_dir itself has the files (flat structure)
                elif collection_dir.name.startswith('l') and collection_dir.name.count('_') >= 1:
                    # Looks like a scene ID, check collection_dir directly
                    if (collection_dir / "seenable_obj_dict.json").exists() and (collection_dir / "object_annots.json").exists():
                        scene_paths.append(collection_dir)
                        break  # Only check once per collection if it matches
    
    if max_scenes:
        scene_paths = scene_paths[:max_scenes]
    
    print(f"Found {len(scene_paths)} scenes with both files\n")
    
    all_results = []
    total_statistics = {
        'total_scenes': len(scene_paths),
        'total_seenable_objects': 0,
        'total_annots_objects': 0,
        'total_seenable_in_annots': 0,
        'total_annots_in_seenable': 0,
        'total_missing_in_seenable': 0,
        'total_missing_in_annots': 0,
        'total_unmapped_seenable': 0,
        'total_unmapped_annots': 0,
        'scenes_with_mismatches': 0,
        'scenes_with_unmapped': 0
    }
    
    for scene_path in scene_paths:
        result = analyze_scene_mismatches(scene_path, sm_to_human_mapping)
        all_results.append(result)
        
        # Aggregate statistics
        total_statistics['total_seenable_objects'] += result['seenable_obj_count']
        total_statistics['total_annots_objects'] += result['object_annots_count']
        total_statistics['total_seenable_in_annots'] += result['seenable_in_annots']
        total_statistics['total_annots_in_seenable'] += result['annots_in_seenable']
        total_statistics['total_missing_in_seenable'] += len(result['missing_in_seenable'])
        total_statistics['total_missing_in_annots'] += len(result['missing_in_annots'])
        total_statistics['total_unmapped_seenable'] += len(result['unmapped_seenable'])
        total_statistics['total_unmapped_annots'] += len(result['unmapped_annots'])
        
        if result['missing_in_seenable'] or result['missing_in_annots']:
            total_statistics['scenes_with_mismatches'] += 1
        if result['unmapped_seenable'] or result['unmapped_annots']:
            total_statistics['scenes_with_unmapped'] += 1
    
    # Print summary
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total scenes analyzed: {total_statistics['total_scenes']}")
    print(f"Total objects in seenable_obj_dict: {total_statistics['total_seenable_objects']}")
    print(f"Total objects in object_annots: {total_statistics['total_annots_objects']}")
    print(f"\nOverlap:")
    print(f"  Objects in seenable that are in annots: {total_statistics['total_seenable_in_annots']} ({total_statistics['total_seenable_in_annots']/max(total_statistics['total_seenable_objects'],1)*100:.1f}%)")
    print(f"  Objects in annots that are in seenable: {total_statistics['total_annots_in_seenable']} ({total_statistics['total_annots_in_seenable']/max(total_statistics['total_annots_objects'],1)*100:.1f}%)")
    print(f"\nMismatches:")
    print(f"  Objects in seenable but NOT in annots: {total_statistics['total_missing_in_annots']}")
    print(f"  Objects in annots but NOT in seenable: {total_statistics['total_missing_in_seenable']}")
    print(f"  Scenes with mismatches: {total_statistics['scenes_with_mismatches']} ({total_statistics['scenes_with_mismatches']/max(total_statistics['total_scenes'],1)*100:.1f}%)")
    print(f"\nMapping issues:")
    print(f"  Unmapped objects in seenable: {total_statistics['total_unmapped_seenable']}")
    print(f"  Unmapped objects in annots: {total_statistics['total_unmapped_annots']}")
    print(f"  Scenes with unmapped objects: {total_statistics['scenes_with_unmapped']} ({total_statistics['scenes_with_unmapped']/max(total_statistics['total_scenes'],1)*100:.1f}%)")
    
    # Find problematic scenes
    print("\n" + "=" * 80)
    print("SCENES WITH MOST ISSUES")
    print("=" * 80)
    
    # Sort by number of issues
    sorted_results = sorted(all_results, 
                          key=lambda x: len(x['missing_in_seenable']) + len(x['missing_in_annots']) + len(x['unmapped_seenable']) + len(x['unmapped_annots']), 
                          reverse=True)
    
    print("\nTop 10 scenes with most missing objects:")
    for result in sorted_results[:10]:
        issues = len(result['missing_in_seenable']) + len(result['missing_in_annots']) + len(result['unmapped_seenable']) + len(result['unmapped_annots'])
        if issues > 0:
            print(f"\n{result['scene_id']}:")
            print(f"  Seenable objects: {result['seenable_obj_count']}, Annots objects: {result['object_annots_count']}")
            print(f"  Missing in seenable: {len(result['missing_in_seenable'])}")
            print(f"  Missing in annots: {len(result['missing_in_annots'])}")
            print(f"  Unmapped (seenable): {len(result['unmapped_seenable'])}")
            print(f"  Unmapped (annots): {len(result['unmapped_annots'])}")
            if result['missing_in_seenable'][:3]:
                print(f"  Sample missing in seenable: {result['missing_in_seenable'][:3]}")
            if result['missing_in_annots'][:3]:
                print(f"  Sample missing in annots: {result['missing_in_annots'][:3]}")
            if result['unmapped_seenable'][:3]:
                print(f"  Sample unmapped (seenable): {result['unmapped_seenable'][:3]}")
            if result['unmapped_annots'][:3]:
                print(f"  Sample unmapped (annots): {result['unmapped_annots'][:3]}")
    
    # Pattern analysis
    print("\n" + "=" * 80)
    print("NAMING PATTERN ANALYSIS")
    print("=" * 80)
    
    all_missing_seenable = []
    all_missing_annots = []
    for result in all_results:
        all_missing_seenable.extend(result['missing_in_seenable'])
        all_missing_annots.extend(result['missing_in_annots'])
    
    if all_missing_seenable:
        print(f"\nObjects in annots but NOT in seenable (sample of {min(20, len(all_missing_seenable))}):")
        for obj_id in list(set(all_missing_seenable))[:20]:
            print(f"  {obj_id}")
    
    if all_missing_annots:
        print(f"\nObjects in seenable but NOT in annots (sample of {min(20, len(all_missing_annots))}):")
        for obj_id in list(set(all_missing_annots))[:20]:
            print(f"  {obj_id}")
    
    return {
        'statistics': total_statistics,
        'scene_results': all_results
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_sim_name_mismatches.py <images_dir> [max_scenes]")
        print("Example: python analyze_sim_name_mismatches.py /path/to/sim_images/jiawei 100")
        sys.exit(1)
    
    images_dir = Path(sys.argv[1])
    max_scenes = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not images_dir.exists():
        print(f"ERROR: Directory not found: {images_dir}")
        sys.exit(1)
    
    analyze_all_scenes(images_dir, max_scenes)

