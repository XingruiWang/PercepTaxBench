#!/usr/bin/env python3
"""
Script to aggregate questions from JSON files into TSV format.
Output format: index	image	question	answer	image_index	level
"""

import json
import os
from pathlib import Path
from PIL import Image
import base64
import io
import os
import os.path as osp

# Input and output paths
input_dir = "/path/to/SpatialReasonerDataGen/qa_gen/visual_qa_results"
output_file = "/path/to/Taxonomy/question_geneation/sampe_taxonomy_questions.tsv"


def resize_image_by_factor(img, factor=1):
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    img = img.resize((new_w, new_h))
    return img

def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    max_size = os.environ.get('VLMEVAL_MAX_IMAGE_SIZE', 1e9)
    min_edge = os.environ.get('VLMEVAL_MIN_IMAGE_EDGE', 1e2)
    max_size = int(max_size)
    min_edge = int(min_edge)
    if min(img.size) < min_edge:
        factor = min_edge / min(img.size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    factor = 1
    while len(ret) > max_size:
        factor *= 0.7  # Half Pixels Per Resize, approximately
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    if factor < 1:
        new_w, new_h = image_new.size
        print(
            f'Warning: image size is too large and exceeds `VLMEVAL_MAX_IMAGE_SIZE` {max_size}, '
            f'resize to {factor:.2f} of original size: ({new_w}, {new_h})'
        )

    return ret

def encode_image_file_to_base64(image_path, target_size=-1, fmt='JPEG'):
    image = Image.open(image_path)
    return encode_image_to_base64(image, target_size=target_size, fmt=fmt)


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P', 'LA'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    base_dir = osp.dirname(image_path)
    if not osp.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    image.save(image_path)
    
def aggregate_questions():
    """Aggregate all questions from JSON files into a TSV file."""
    all_questions = []
    index = 0
    
    # Get all JSON files except the summary
    json_files = sorted([f for f in os.listdir(input_dir) 
                        if f.endswith('_qa.json') and f != 'qa_generation_summary.json'])
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        file_path = os.path.join(input_dir, json_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_path = data.get('image_path', '')
            image_path = os.path.join(input_dir, '../..', image_path)
            
            # Extract scene and frame from filename (e.g., s0_f1_qa.json -> s0, f1)
            parts = json_file.replace('_qa.json', '').split('_')

            questions = data.get('questions', [])
            print(f"Processing {json_file}: {len(questions)} questions")
            
            for q in questions:
                question_text = q.get('question', '').strip()
                answer_text = q.get('answer', '').strip()
                
                # Clean up text - replace tabs and newlines to preserve TSV format
                question_text = question_text.replace('\t', ' ').replace('\n', ' ')
                answer_text = answer_text.replace('\t', ' ').replace('\n', ' ')
                
                all_questions.append({
                    'index': index,
                    'image': encode_image_file_to_base64(image_path, target_size=512),
                    'question': question_text,
                    'answer': answer_text,
                })
                index += 1
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Write to TSV file
    print(f"\nWriting {len(all_questions)} questions to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("index\timage\tquestion\tanswer\n")
        
        # Write data
        for q in all_questions:
            f.write(f"{q['index']}\t{q['image']}\t{q['question']}\t{q['answer']}\n")
    
    print(f"Done! Total questions aggregated: {len(all_questions)}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Total files processed: {len(json_files)}")
    print(f"  Total questions: {len(all_questions)}")
    

if __name__ == "__main__":
    aggregate_questions()
