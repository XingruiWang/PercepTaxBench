#!/usr/bin/env python3
"""
Script to aggregate questions from unified QA JSON files into TSV format for VLMEvalKit.
Output format: index	image	question	answer

This unified script works for both real and sim images:
- Input: all_questions.json + images/ structure
- Output: TSV format compatible with VLMEvalKit taxonomy benchmark
- Automatically handles both image_id (real images) and image_path (sim images) formats
- Uses bounding box images (bbox.jpg) with fallback to original.jpg for real images
- Encodes images to base64 for VLMEvalKit compatibility
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image
import base64
import io
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resize_image_by_factor(img, factor=1):
    """Resize image by a given factor."""
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    img = img.resize((new_w, new_h))
    return img

def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    """
    Encode PIL Image to base64 string with optional resizing.
    
    Args:
        img: PIL Image object
        target_size: Maximum size for image (if > 0, will resize)
        fmt: Image format for encoding
    
    Returns:
        base64 encoded string of the image
    """
    # Convert RGBA/P/LA to RGB if needed
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    
    # Resize if target_size is specified
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    
    # Save to buffer
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    
    # Handle size constraints from environment variables
    max_size = os.environ.get('VLMEVAL_MAX_IMAGE_SIZE', 1e9)
    min_edge = os.environ.get('VLMEVAL_MIN_IMAGE_EDGE', 1e2)
    max_size = int(max_size)
    min_edge = int(min_edge)
    
    # Ensure minimum edge size
    if min(img.size) < min_edge:
        factor = min_edge / min(img.size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')
    
    # Reduce size if too large
    factor = 1
    while len(ret) > max_size:
        factor *= 0.7  # Reduce by ~30% each iteration
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')
    
    if factor < 1:
        new_w, new_h = image_new.size
        logger.warning(
            f'Image size exceeded VLMEVAL_MAX_IMAGE_SIZE {max_size}, '
            f'resized to {factor:.2f} of original size: ({new_w}, {new_h})'
        )
    
    return ret

def encode_image_file_to_base64(image_path, target_size=-1, fmt='JPEG'):
    """Load image from file and encode to base64."""
    try:
        image = Image.open(image_path)
        return encode_image_to_base64(image, target_size=target_size, fmt=fmt)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def aggregate_unified_qa_questions(input_dir, output_file, target_image_size=512):
    """
    Aggregate all questions from unified QA JSON file into a TSV file.
    This script works for both real and sim images.
    
    Args:
        input_dir: Directory containing taxonomyQABench_realimage/ or taxonomyQABench_simimage/ folder
        output_file: Output TSV file path
        target_image_size: Target size for image encoding
    """
    
    # Construct paths - try both real and sim image directories
    unified_qa_dir = None
    for dir_name in ["taxonomyQABench_realimage", "taxonomyQABench_simimage"]:
        test_dir = os.path.join(input_dir, dir_name)
        if os.path.exists(test_dir):
            unified_qa_dir = test_dir
            break
    
    # If not found, assume input_dir itself is the benchmark directory
    if not unified_qa_dir:
        if os.path.exists(os.path.join(input_dir, "all_questions.json")):
            unified_qa_dir = input_dir
        else:
            raise FileNotFoundError(f"Neither taxonomyQABench_realimage nor taxonomyQABench_simimage found in {input_dir}, and input_dir doesn't contain all_questions.json")
    
    questions_file = os.path.join(unified_qa_dir, "all_questions.json")
    images_dir = os.path.join(unified_qa_dir, "images")
    
    # Validate input paths
    if not os.path.exists(unified_qa_dir):
        raise FileNotFoundError(f"Unified QA directory not found: {unified_qa_dir}")
    
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    logger.info(f"Processing questions from: {questions_file}")
    logger.info(f"Images directory: {images_dir}")
    
    # Load questions
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading questions file: {e}")
        return
    
    logger.info(f"Loaded {len(questions_data)} questions")
    
    all_questions = []
    processed_count = 0
    skipped_count = 0
    
    for question_data in questions_data:
        try:
            # Extract question information
            # Note: Using question/answer (with color box references) for TSV, not question_object/answer_object
            question_index = question_data.get('question_index', processed_count)
            question_text = question_data.get('question', '').strip()
            answer_text = question_data.get('answer', '').strip()
            image_path_field = question_data.get('image_path', '')
            image_id = question_data.get('image_id', '')
            
            # Skip if essential fields are missing
            if not question_text or not answer_text:
                logger.warning(f"Skipping question {question_index}: missing essential fields")
                skipped_count += 1
                continue
            
            # Determine image path - handle both real and sim image formats
            final_image_path = None
            
            if image_path_field:
                # Use image_path directly (simpler, works for sim images and most real images)
                full_path = os.path.join(images_dir, image_path_field)
                if os.path.exists(full_path):
                    final_image_path = full_path
                else:
                    # For real images: image_path might be like "d1eb04f4822707bd/bbox.jpg"
                    # Try the direct path first, then fallback
                    logger.debug(f"image_path not found: {full_path}, trying fallback")
                    # Will fall through to image_id construction below
                    pass
            
            # If no image_path or it failed, construct from image_id (for real images)
            if not final_image_path and image_id:
                bbox_image_path = os.path.join(images_dir, image_id, "bbox.jpg")
                original_image_path = os.path.join(images_dir, image_id, "original.jpg")
                
                # Prefer bounding box image, fallback to original if not available
                if os.path.exists(bbox_image_path):
                    final_image_path = bbox_image_path
                    logger.debug(f"Using bounding box image: {final_image_path}")
                elif os.path.exists(original_image_path):
                    final_image_path = original_image_path
                    logger.warning(f"Bounding box image not found, using original: {final_image_path}")
            
            if not final_image_path:
                logger.warning(f"No image found for question {question_index}")
                skipped_count += 1
                continue
            
            image_path = final_image_path
            
            # Encode image to base64
            image_base64 = encode_image_file_to_base64(image_path, target_size=target_image_size)
            if image_base64 is None:
                logger.warning(f"Failed to encode image: {image_path}")
                skipped_count += 1
                continue
            
            # Clean up text - replace tabs and newlines to preserve TSV format
            question_text = question_text.replace('\t', ' ').replace('\n', ' ')
            answer_text = answer_text.replace('\t', ' ').replace('\n', ' ')
            
            # Add to results
            all_questions.append({
                'index': question_index,
                'image': image_base64,
                'question': question_text,
                'answer': answer_text,
                'image_id': image_id
            })
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} questions...")
                
        except Exception as e:
            logger.error(f"Error processing question {question_data.get('question_index', 'unknown')}: {e}")
            skipped_count += 1
            continue
    
    # Write to TSV file
    logger.info(f"Writing {len(all_questions)} questions to {output_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("index\timage\tquestion\tanswer\n")
            
            # Write data
            for q in all_questions:
                f.write(f"{q['index']}\t{q['image']}\t{q['question']}\t{q['answer']}\n")
        
        logger.info(f"Successfully wrote TSV file: {output_file}")
        
    except Exception as e:
        logger.error(f"Error writing TSV file: {e}")
        return
    
    # Print statistics
    logger.info("=" * 60)
    logger.info("AGGREGATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total questions processed: {processed_count}")
    logger.info(f"Questions skipped: {skipped_count}")
    logger.info(f"Success rate: {(processed_count / (processed_count + skipped_count) * 100):.1f}%")
    logger.info(f"Output file: {output_file}")
    
    return processed_count

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Aggregate unified QA questions into TSV format for VLMEvalKit (works for both real and sim images)')
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='/path/to/SpatialReasonerDataGen/qa_gen',
        help='Directory containing taxonomyQABench_realimage/ or taxonomyQABench_simimage/ folder'
    )
    
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='/path/to/SpatialReasonerDataGen/qa_gen/taxonomy_benchmark.tsv',
        help='Output TSV file path'
    )
    
    parser.add_argument(
        '--target_image_size', 
        type=int, 
        default=512,
        help='Target size for image encoding (default: 512)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting unified QA aggregation...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Target image size: {args.target_image_size}")
    
    try:
        count = aggregate_unified_qa_questions(
            input_dir=args.input_dir,
            output_file=args.output_file,
            target_image_size=args.target_image_size
        )
        
        if count > 0:
            logger.info(f" Successfully processed {count} questions!")
        else:
            logger.error(" No questions were processed successfully")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f" Aggregation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
