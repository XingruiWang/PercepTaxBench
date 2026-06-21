#!/usr/bin/env python3
"""
Analyze Taxonomy QA Benchmark results for anomalies we fixed:
1. Spatial questions with ambiguous distances (objects too close)
2. Images with >6 objects (should be skipped)
3. Questions with limited choices (should include all objects)
4. Questions on void clusters (humans, animals) - should be filtered
5. Description matching conflicts - should be filtered
6. Question count per image (should be ~10-12, not ~22)
7. Material variant questions (should be 0, only material_property)
8. Spatial question count per image (should be 1-2)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

sys.path.append('/path/to/SpatialReasonerDataGen/qa_gen/scripts')

def load_questions(json_file: Path) -> List[Dict[str, Any]]:
    """Load questions from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get('questions', [])
    else:
        return []

def analyze_qa_anomalies(questions_file: Path):
    """Analyze questions for various anomalies"""
    
    print("=" * 80)
    print("TAXONOMY QA BENCHMARK ANOMALY ANALYSIS")
    print("=" * 80)
    print()
    
    if not questions_file.exists():
        print(f"ERROR: File not found: {questions_file}")
        return
    
    print(f"Loading questions from: {questions_file}")
    questions = load_questions(questions_file)
    print(f"Total questions loaded: {len(questions)}")
    print()
    
    # Group questions by image_id
    questions_by_image = defaultdict(list)
    for q in questions:
        image_id = q.get('image_id', 'unknown')
        questions_by_image[image_id].append(q)
    
    print(f"Total images: {len(questions_by_image)}")
    print()
    
    # 1. Check question count per image
    print("=" * 80)
    print("1. QUESTION COUNT PER IMAGE")
    print("=" * 80)
    question_counts = [len(qs) for qs in questions_by_image.values()]
    if question_counts:
        avg_questions = sum(question_counts) / len(question_counts)
        max_questions = max(question_counts)
        min_questions = min(question_counts)
        print(f"Average questions per image: {avg_questions:.2f} (target: ~10-12)")
        print(f"Min questions per image: {min_questions}")
        print(f"Max questions per image: {max_questions} (should be <= 15)")
        print(f"Images with >15 questions: {sum(1 for c in question_counts if c > 15)}")
        print(f"Images with >20 questions: {sum(1 for c in question_counts if c > 20)}")
    print()
    
    # 2. Check object count per image (should be <= 6)
    print("=" * 80)
    print("2. OBJECT COUNT PER IMAGE")
    print("=" * 80)
    object_counts = []
    images_with_too_many_objects = []
    for image_id, qs in questions_by_image.items():
        if qs:
            # Get unique objects from all questions
            all_objects = set()
            for q in qs:
                objects = q.get('objects', q.get('choices', []))
                if objects:
                    all_objects.update(objects if isinstance(objects, list) else [objects])
            obj_count = len(all_objects)
            object_counts.append(obj_count)
            if obj_count > 6:
                images_with_too_many_objects.append((image_id, obj_count))
    
    if object_counts:
        avg_objects = sum(object_counts) / len(object_counts)
        max_objects = max(object_counts)
        print(f"Average objects per image: {avg_objects:.2f}")
        print(f"Max objects per image: {max_objects} (should be <= 6)")
        print(f"Images with >6 objects: {len(images_with_too_many_objects)}")
        if images_with_too_many_objects[:5]:
            print("Sample images with >6 objects:")
            for img_id, count in images_with_too_many_objects[:5]:
                print(f"  {img_id}: {count} objects")
    print()
    
    # 3. Check choices vs available objects
    print("=" * 80)
    print("3. CHOICES LIMITATION CHECK")
    print("=" * 80)
    limited_choices_issues = []
    for image_id, qs in questions_by_image.items():
        if not qs:
            continue
        
        # Get all available objects
        all_available_objects = set()
        for q in qs:
            objects = q.get('objects', [])
            if objects:
                all_available_objects.update(objects if isinstance(objects, list) else [objects])
        
        # Check each question
        for q in qs:
            choices = q.get('choices', [])
            objects = q.get('objects', [])
            
            if not choices or not isinstance(choices, list):
                continue
            
            available_count = len(all_available_objects)
            choices_count = len(choices)
            
            # Non-spatial questions should include all available objects
            question_type = q.get('question_type', '')
            if not question_type.startswith('spatial_'):
                if choices_count < available_count:
                    limited_choices_issues.append({
                        'image_id': image_id,
                        'question_type': question_type,
                        'available': available_count,
                        'choices': choices_count,
                        'question': q.get('question', '')[:60]
                    })
    
    print(f"Questions with limited choices (should include all objects): {len(limited_choices_issues)}")
    if limited_choices_issues[:5]:
        print("Sample issues:")
        for issue in limited_choices_issues[:5]:
            print(f"  Image {issue['image_id']}: {issue['available']} available, {issue['choices']} choices ({issue['question_type']})")
            print(f"    Question: {issue['question']}...")
    print()
    
    # 4. Check material question types
    print("=" * 80)
    print("4. MATERIAL QUESTION TYPES")
    print("=" * 80)
    material_question_types = defaultdict(int)
    for q in questions:
        qtype = q.get('question_type', '')
        if 'material' in qtype.lower():
            material_question_types[qtype] += 1
    
    print(f"Total material-related questions: {sum(material_question_types.values())}")
    print("Breakdown by type:")
    for qtype, count in sorted(material_question_types.items(), key=lambda x: -x[1]):
        print(f"  {qtype}: {count}")
    
    material_variants = [qtype for qtype in material_question_types.keys() 
                        if qtype.startswith('material_') and qtype != 'material_property']
    if material_variants:
        print(f"\nWARNING: Found {len(material_variants)} material variant types (should be 0):")
        for variant in material_variants:
            print(f"  {variant}: {material_question_types[variant]}")
    else:
        print("\n✓ No material variants found (only material_property)")
    print()
    
    # 5. Check spatial question count per image
    print("=" * 80)
    print("5. SPATIAL QUESTION COUNT PER IMAGE")
    print("=" * 80)
    spatial_counts = []
    for image_id, qs in questions_by_image.items():
        spatial_q_count = sum(1 for q in qs if q.get('question_type', '').startswith('spatial_'))
        if spatial_q_count > 0:
            spatial_counts.append(spatial_q_count)
    
    if spatial_counts:
        avg_spatial = sum(spatial_counts) / len(spatial_counts) if spatial_counts else 0
        max_spatial = max(spatial_counts) if spatial_counts else 0
        print(f"Average spatial questions per image: {avg_spatial:.2f} (target: 1-2)")
        print(f"Max spatial questions per image: {max_spatial} (target: <= 2)")
        print(f"Images with >2 spatial questions: {sum(1 for c in spatial_counts if c > 2)}")
    print()
    
    # 6. Check for void cluster objects (human, animal, etc.)
    print("=" * 80)
    print("6. VOID CLUSTER OBJECTS CHECK")
    print("=" * 80)
    void_cluster_keywords = ['human', 'person', 'people', 'man', 'woman', 'child', 'adult',
                            'animal', 'dog', 'cat', 'bird', 'horse', 'cow', 'pig']
    
    void_cluster_questions = []
    for q in questions:
        question_text = q.get('question', '').lower()
        answer = str(q.get('answer', '')).lower()
        question_type = q.get('question_type', '')
        
        # Check if question involves void cluster objects
        if any(keyword in question_text or keyword in answer for keyword in void_cluster_keywords):
            # Skip spatial questions (they can involve humans/animals)
            if not question_type.startswith('spatial_'):
                void_cluster_questions.append({
                    'question_type': question_type,
                    'question': q.get('question', '')[:80],
                    'image_id': q.get('image_id', '')
                })
    
    print(f"Non-spatial questions potentially involving void clusters: {len(void_cluster_questions)}")
    if void_cluster_questions[:5]:
        print("Sample questions (manual review needed):")
        for vq in void_cluster_questions[:5]:
            print(f"  [{vq['question_type']}] {vq['question']}... (Image: {vq['image_id']})")
    print()
    
    # 7. Check description_matching questions for conflicts
    print("=" * 80)
    print("7. DESCRIPTION MATCHING QUESTIONS")
    print("=" * 80)
    desc_matching_questions = [q for q in questions if q.get('question_type') == 'description_matching']
    print(f"Total description_matching questions: {len(desc_matching_questions)}")
    
    # Check for ambiguous patterns (same material/function mentioned in question)
    ambiguous_desc_questions = []
    for q in desc_matching_questions:
        question_text = q.get('question', '').lower()
        answer = str(q.get('answer', '')).lower()
        
        # Extract description from question (text after "description:")
        if 'description:' in question_text:
            desc_part = question_text.split('description:')[1].strip().strip("'\"")
            # If description mentions material/function that could match multiple objects
            if 'made of' in desc_part or 'used for' in desc_part:
                ambiguous_desc_questions.append({
                    'question': q.get('question', '')[:80],
                    'image_id': q.get('image_id', '')
                })
    
    print(f"Potentially ambiguous description_matching questions: {len(ambiguous_desc_questions)}")
    if ambiguous_desc_questions[:3]:
        print("Sample questions (check for conflicts):")
        for aq in ambiguous_desc_questions[:3]:
            print(f"  {aq['question']}... (Image: {aq['image_id']})")
    print()
    
    # 8. Check spatial questions for "unknown" answers (should be filtered)
    print("=" * 80)
    print("8. SPATIAL QUESTIONS - UNKNOWN ANSWERS")
    print("=" * 80)
    spatial_unknown = [q for q in questions 
                       if q.get('question_type', '').startswith('spatial_') 
                       and str(q.get('answer', '')).lower() == 'unknown']
    print(f"Spatial questions with 'unknown' answer (should be filtered): {len(spatial_unknown)}")
    if spatial_unknown[:3]:
        print("Sample questions with unknown answers:")
        for sq in spatial_unknown[:3]:
            print(f"  [{sq.get('question_type')}] {sq.get('question', '')[:60]}... (Image: {sq.get('image_id')})")
    print()
    
    # 9. Question type distribution
    print("=" * 80)
    print("9. QUESTION TYPE DISTRIBUTION")
    print("=" * 80)
    question_type_counts = defaultdict(int)
    for q in questions:
        qtype = q.get('question_type', 'unknown')
        question_type_counts[qtype] += 1
    
    print("Top question types:")
    for qtype, count in sorted(question_type_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {qtype}: {count}")
    print()
    
    # 10. Hard vs Easy questions
    print("=" * 80)
    print("10. HARD VS EASY QUESTIONS")
    print("=" * 80)
    hard_prefixes = ['repurposing_', 'counterfactual_', 'compositional_', 'latent_']
    hard_count = sum(1 for q in questions 
                    if any(q.get('question_type', '').startswith(prefix) for prefix in hard_prefixes))
    easy_count = len(questions) - hard_count
    
    print(f"Hard questions (repurposing, counterfactual, compositional, latent): {hard_count}")
    print(f"Easy questions: {easy_count}")
    print(f"Hard question percentage: {(hard_count/len(questions)*100):.1f}%" if questions else "N/A")
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_qa_anomalies.py <questions.json>")
        print("Example: python analyze_qa_anomalies.py ../../taxonomyQABench_realimage/all_questions.json")
        sys.exit(1)
    
    questions_file = Path(sys.argv[1])
    analyze_qa_anomalies(questions_file)

