#!/usr/bin/env python3
"""
Analyze mismatches between detected objects and RAM tags in OpenImages unified output.
Generates refinement mappings for better object naming.
"""

import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DetectionTagAnalyzer:
    def __init__(self, unified_output_dir: str, object_descriptions_dir: str):
        self.unified_output_dir = Path(unified_output_dir)
        self.object_descriptions_dir = Path(object_descriptions_dir)
        
        self.detected_classes = Counter()
        self.tag_words = Counter()
        self.refinement_opportunities = defaultdict(list)
        self.known_objects = self.load_known_objects()
        
    def load_known_objects(self) -> Set[str]:
        """Load all known objects from object description files"""
        objects = set()
        
        # Load from sm_objects_138.txt
        sm_objects_file = self.object_descriptions_dir / "sm_objects_138.txt"
        if sm_objects_file.exists():
            with open(sm_objects_file) as f:
                for line in f:
                    obj = line.strip()
                    if obj:
                        objects.add(obj.lower())
            logger.info(f"Loaded {len(objects)} objects from sm_objects_138.txt")
        
        # Load from original_openimages_objects.txt
        openimages_file = self.object_descriptions_dir / "original_openimages_objects.txt"
        if openimages_file.exists():
            with open(openimages_file) as f:
                for line in f:
                    obj = line.strip()
                    if obj:
                        objects.add(obj.lower())
            logger.info(f"Total {len(objects)} objects after loading original_openimages_objects.txt")
        
        logger.info(f"Loaded {len(objects)} known objects total")
        return objects
    
    def analyze_file(self, annotation_file: Path) -> Dict:
        """Analyze a single annotation file"""
        with open(annotation_file) as f:
            data = json.load(f)
        
        tags = [t.lower() for t in data.get('tags', [])]
        detections = data.get('detections', [])
        
        self.tag_words.update(tags)
        
        refinements = []
        for detection in detections:
            class_name = detection['class_name'].lower()
            self.detected_classes[class_name] += 1
            
            # Check for potential refinements
            better_alternatives = self.find_better_alternatives(class_name, tags)
            if better_alternatives:
                refinements.append({
                    'detected': class_name,
                    'alternatives': better_alternatives,
                    'tags': tags[:20]
                })
        
        return {
            'file': annotation_file.stem,
            'num_detections': len(detections),
            'num_tags': len(tags),
            'refinements': refinements
        }
    
    def find_better_alternatives(self, detected_class: str, available_tags: List[str]) -> List[str]:
        """Find better alternatives for a detected class from available tags"""
        alternatives = []
        
        generic_to_specific = {
            'army': ['soldier', 'military personnel', 'guard', 'serviceman'],
            'person': ['man', 'woman', 'child', 'boy', 'girl', 'worker', 'pedestrian'],
            'building': ['house', 'church', 'monument', 'store', 'office', 'temple'],
            'vehicle': ['car', 'truck', 'van', 'bus', 'motorcycle', 'bicycle'],
            'animal': ['dog', 'cat', 'horse', 'bird', 'elephant', 'cow'],
            'food': ['pizza', 'burger', 'sandwich', 'cake', 'bread', 'fruit'],
            'furniture': ['chair', 'table', 'sofa', 'bed', 'desk', 'cabinet'],
            'plant': ['tree', 'flower', 'bush', 'grass', 'shrub'],
            'weapon': ['gun', 'rifle', 'sword', 'knife', 'spear'],
            'clothing': ['shirt', 'pants', 'dress', 'coat', 'shoes', 'hat'],
        }
        
        if detected_class in generic_to_specific:
            for better_tag in generic_to_specific[detected_class]:
                if better_tag in available_tags:
                    # Check if better tag is in known objects
                    if better_tag in self.known_objects:
                        alternatives.append(better_tag)
        
        return alternatives
    
    def analyze_all(self, max_files: int = None) -> Dict:
        """Analyze all annotation files"""
        # Files are in subdirectories: <id>/annotations/<id>.json
        annotation_files = list(self.unified_output_dir.glob("*/annotations/*.json"))
        
        if max_files:
            annotation_files = annotation_files[:max_files]
        
        logger.info(f"Analyzing {len(annotation_files)} files...")
        
        files_with_refinements = []
        total_refinements = 0
        
        for i, ann_file in enumerate(annotation_files):
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(annotation_files)} files...")
            
            result = self.analyze_file(ann_file)
            if result['refinements']:
                files_with_refinements.append(result)
                total_refinements += len(result['refinements'])
                
                # Track refinement opportunities
                for ref in result['refinements']:
                    key = f"{ref['detected']} → {ref['alternatives'][0]}"
                    self.refinement_opportunities[key].append(result['file'])
        
        logger.info(f"Analysis complete!")
        logger.info(f"Files with refinement opportunities: {len(files_with_refinements)}")
        logger.info(f"Total refinements found: {total_refinements}")
        
        return {
            'total_files': len(annotation_files),
            'files_with_refinements': len(files_with_refinements),
            'total_refinements': total_refinements,
            'detected_classes': dict(self.detected_classes.most_common(50)),
            'tag_words': dict(self.tag_words.most_common(50)),
            'refinement_opportunities': {k: len(v) for k, v in self.refinement_opportunities.items()},
            'sample_refinements': files_with_refinements[:20]
        }
    
    def generate_refined_mappings(self) -> Dict:
        """Generate refined mappings based on analysis"""
        mappings = {
            'generic_to_specific': {},
            'refinement_statistics': {}
        }
        
        # Count how often each refinement appears
        for refinement, files in self.refinement_opportunities.items():
            detected, better = refinement.split(' → ')
            count = len(files)
            
            if count >= 5:  # Only include if seen in 5+ files
                if detected not in mappings['generic_to_specific']:
                    mappings['generic_to_specific'][detected] = []
                
                # Check if better alternative exists in known objects
                if better in self.known_objects:
                    mappings['generic_to_specific'][detected].append(better)
                    mappings['refinement_statistics'][refinement] = {
                        'count': count,
                        'in_known_objects': True
                    }
        
        return mappings


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze detection-tag mismatches')
    parser.add_argument('--unified_dir', type=str, 
                       default='/path/to/project/openimages_unified_output',
                       help='Path to unified output directory')
    parser.add_argument('--object_descriptions_dir', type=str,
                       default='../object_descriptions',
                       help='Path to object descriptions directory')
    parser.add_argument('--output', type=str,
                       default='detection_tag_analysis.json',
                       help='Output JSON file')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum files to analyze (None = all)')
    
    args = parser.parse_args()
    
    analyzer = DetectionTagAnalyzer(args.unified_dir, args.object_descriptions_dir)
    results = analyzer.analyze_all(max_files=args.max_files)
    
    # Save analysis results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Analysis saved to {args.output}")
    
    # Generate refined mappings
    refined_mappings = analyzer.generate_refined_mappings()
    mappings_file = args.output.replace('.json', '_refined_mappings.json')
    
    with open(mappings_file, 'w') as f:
        json.dump(refined_mappings, f, indent=2)
    
    logger.info(f"Refined mappings saved to {mappings_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("DETECTION-TAG MISMATCH ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal files analyzed: {results['total_files']}")
    print(f"Files with refinement opportunities: {results['files_with_refinements']}")
    print(f"Total refinements found: {results['total_refinements']}")
    
    print("\n" + "=" * 80)
    print("TOP DETECTED CLASSES")
    print("=" * 80)
    for obj, count in list(results['detected_classes'].items())[:20]:
        print(f"  {obj:30s} - {count:5d} detections")
    
    print("\n" + "=" * 80)
    print("REFINEMENT OPPORTUNITIES (sorted by frequency)")
    print("=" * 80)
    sorted_refinements = sorted(results['refinement_opportunities'].items(), 
                               key=lambda x: x[1], reverse=True)
    for refinement, count in sorted_refinements[:20]:
        print(f"  {refinement:50s} - {count:4d} cases")
    
    print("\n" + "=" * 80)
    print("GENERATED REFINED MAPPINGS")
    print("=" * 80)
    for detected, alternatives in refined_mappings['generic_to_specific'].items():
        print(f"  {detected:20s} → {', '.join(alternatives)}")
    
    print("\n" + "=" * 80)
    print("SAMPLE CASES WITH REFINEMENTS")
    print("=" * 80)
    for sample in results['sample_refinements'][:5]:
        print(f"\nFile: {sample['file']}")
        for ref in sample['refinements'][:3]:
            print(f"  Detected: '{ref['detected']}' → Better: {ref['alternatives']}")
            print(f"  Tags: {', '.join(ref['tags'][:10])}...")


if __name__ == '__main__':
    main()

