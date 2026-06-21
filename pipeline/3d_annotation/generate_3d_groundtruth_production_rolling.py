#!/usr/bin/env python3
"""
Generate 3D ground truth using full pipeline with rolling statistics and checkpointing.
This version implements robust logging that handles crashes and timeouts gracefully.
"""

import os
import sys
import json
import logging
import hashlib
import argparse
import time
from typing import List, Dict, Any
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Add the parent directory to the path to import srdatagen modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the new 3d_annotation folder
from config import cfg
from reconstruct3d import Reconstruct3D
from pose3d_orientanything import Pose3DOrientAnything
from visualize_3d_data import visualize_3d_data

# Import from srdatagen modules that remain in the original location
from srdatagen.modules import TagAndSegment
import srdatagen.utils


def parse_args():
    parser = argparse.ArgumentParser(description='Generate 3D ground truth with rolling statistics and checkpointing.')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to directory containing images')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to output directory for results')
    parser.add_argument('--stats_path', type=str, required=True,
                       help='Path to directory for rolling statistics')
    parser.add_argument('--range_low', type=int, default=None,
                       help='Start index for image range')
    parser.add_argument('--range_high', type=int, default=None,
                       help='End index for image range')
    parser.add_argument('--md5', type=str, default=None,
                       help='MD5 hash for image list validation')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0, cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Number of images to process in each batch')
    parser.add_argument('--max_workers', type=int, default=8,
                       help='Maximum number of parallel workers for I/O operations')
    parser.add_argument('--save_pcd', action='store_true',
                       help='Save full point cloud data')
    parser.add_argument('--enable_pose3d', action='store_true',
                       help='Enable 3D pose estimation')
    parser.add_argument('--enable_pose_filtering', action='store_true',
                       help='Enable filtering to only keep detections with valid poses and bounding boxes above threshold')
    parser.add_argument('--generate_annotations', action='store_true',
                       help='Generate annotated images with bounding boxes and 3D poses')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--skip_visualizations', action='store_true',
                       help='Skip 3D visualizations for maximum speed')
    parser.add_argument('--skip_object_crops', action='store_true',
                       help='Skip object crop generation for maximum speed')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                       help='Save rolling statistics every N images')
    parser.add_argument('--resume_from_checkpoint', action='store_true',
                       help='Resume processing from last checkpoint')
    return parser.parse_args()


def setup_logging_with_level(level: str):
    """Setup logging with specified level"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline_rolling.log')
        ]
    )


def load_checkpoint_stats(stats_path):
    """Load rolling statistics from checkpoint file"""
    checkpoint_file = os.path.join(stats_path, 'rolling_stats_checkpoint.json')
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                stats = json.load(f)
            logging.info(f"Loaded checkpoint with {stats['total_processed']} images processed")
            return stats
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
    
    # Return fresh stats if no checkpoint
    return {
        'total_processed': 0,
        'total_objects_detected': 0,
        'unique_classes': set(),
        'class_counts': {},
        'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
        'pose_statistics': {'with_pose': 0, 'without_pose': 0},
        'processing_times': [],
        'last_checkpoint_time': time.time(),
        'start_time': time.time()
    }


def save_checkpoint_stats(stats, stats_path):
    """Save rolling statistics to checkpoint file"""
    checkpoint_file = os.path.join(stats_path, 'rolling_stats_checkpoint.json')
    
    # Convert set to list for JSON serialization
    stats_copy = stats.copy()
    stats_copy['unique_classes'] = list(stats['unique_classes'])
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(stats_copy, f, indent=2)
        logging.info(f"Checkpoint saved: {stats['total_processed']} images processed")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")


def update_rolling_stats(stats, result):
    """Update rolling statistics with new result"""
    if result and 'detections' in result:
        detections = result['detections']
        stats['total_objects_detected'] += len(detections)
        
        for det in detections:
            # Class statistics
            class_name = det.get('class_name', 'unknown')
            stats['unique_classes'].add(class_name)
            stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
            
            # Confidence distribution
            confidence = det.get('confidence', 0.0)
            if confidence >= 0.8:
                stats['confidence_distribution']['high'] += 1
            elif confidence >= 0.6:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
            
            # Pose statistics
            if 'pcd_orient_bbox' in det and det['pcd_orient_bbox']:
                stats['pose_statistics']['with_pose'] += 1
            else:
                stats['pose_statistics']['without_pose'] += 1
    
    stats['total_processed'] += 1
    
    # Add processing time if available
    if 'processing_time' in result:
        stats['processing_times'].append(result['processing_time'])


def generate_rolling_summary(stats, stats_path):
    """Generate human-readable rolling summary"""
    summary_file = os.path.join(stats_path, 'rolling_summary.txt')
    
    try:
        with open(summary_file, 'w') as f:
            f.write("=== ROLLING PROCESSING STATISTICS ===\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Processing Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Images Processed: {stats['total_processed']}\n")
            f.write(f"Total Objects Detected: {stats['total_objects_detected']}\n")
            f.write(f"Unique Object Classes: {len(stats['unique_classes'])}\n")
            
            if stats['total_processed'] > 0:
                avg_objects_per_image = stats['total_objects_detected'] / stats['total_processed']
                f.write(f"Average Objects per Image: {avg_objects_per_image:.2f}\n")
            
            # Calculate processing rate
            elapsed_time = time.time() - stats['start_time']
            if elapsed_time > 0:
                processing_rate = stats['total_processed'] / (elapsed_time / 3600)  # images per hour
                f.write(f"Processing Rate: {processing_rate:.2f} images/hour\n")
                
            if stats['total_processed'] < 10003 and processing_rate > 0:  # Assuming 10k target
                remaining_images = 10003 - stats['total_processed']
                eta_hours = remaining_images / processing_rate
                f.write(f"Estimated Time Remaining: {eta_hours:.1f} hours\n")
            
            f.write("\nTop 20 Object Classes by Frequency:\n")
            f.write("-" * 40 + "\n")
            sorted_classes = sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True)
            for i, (class_name, count) in enumerate(sorted_classes[:20]):
                f.write(f"{i+1:2d}. {class_name:<20} : {count:>6}\n")
            
            f.write("\nConfidence Distribution:\n")
            f.write("-" * 25 + "\n")
            f.write(f"High (≥0.8): {stats['confidence_distribution']['high']}\n")
            f.write(f"Medium (0.6-0.8): {stats['confidence_distribution']['medium']}\n")
            f.write(f"Low (<0.6): {stats['confidence_distribution']['low']}\n\n")
            
            f.write("Pose Estimation Statistics:\n")
            f.write("-" * 28 + "\n")
            f.write(f"Objects with 3D Pose: {stats['pose_statistics']['with_pose']}\n")
            f.write(f"Objects without 3D Pose: {stats['pose_statistics']['without_pose']}\n")
            
            if stats['total_objects_detected'] > 0:
                pose_rate = (stats['pose_statistics']['with_pose'] / 
                           stats['total_objects_detected'] * 100)
                f.write(f"Pose Success Rate: {pose_rate:.2f}%\n")
        
        logging.info(f"Rolling summary saved to: {summary_file}")
        
    except Exception as e:
        logging.error(f"Failed to generate rolling summary: {e}")


def is_image_processed(image_name, output_path, require_crops=True, require_visualizations=True):
    """Check if an image has already been processed with all required outputs."""
    image_output_dir = os.path.join(output_path, image_name)
    
    if not os.path.exists(image_output_dir) or not os.path.isdir(image_output_dir):
        return False
    
    annotations_dir = os.path.join(image_output_dir, 'annotations')
    if not os.path.exists(annotations_dir):
        return False
    
    if require_crops:
        crops_dir = os.path.join(image_output_dir, 'object_crops')
        if not os.path.exists(crops_dir):
            return False
        crop_files = [f for f in os.listdir(crops_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if len(crop_files) == 0:
            return False
    
    if require_visualizations:
        visualizations_dir = os.path.join(image_output_dir, 'visualizations')
        if not os.path.exists(visualizations_dir):
            return False
        viz_files = [f for f in os.listdir(visualizations_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if len(viz_files) == 0:
            return False
    
    return True


def load_all_images(args, extensions=['.jpg', '.jpeg', '.png']) -> List[str]:
    """Load and validate image list, optionally filtering out already processed images"""
    all_images = [
        x for x in os.listdir(args.image_path)
        if any(x.lower().endswith(ext) for ext in extensions)
    ]
    
    all_images.sort()
    
    # Filter out already processed images if output directory exists
    if os.path.exists(args.output_path):
        unprocessed_images = []
        for img in all_images:
            image_name = os.path.splitext(img)[0]
            if not is_image_processed(image_name, args.output_path, require_crops=True, require_visualizations=True):
                unprocessed_images.append(img)
            else:
                logging.debug(f"Skipping already processed image: {img}")
        
        logging.info(f"Found {len(all_images)} total images, {len(unprocessed_images)} unprocessed")
        all_images = unprocessed_images
    
    if args.range_low is None and args.range_high is None:
        return all_images
    
    if args.md5 is not None:
        md5 = hashlib.md5(','.join(all_images).encode('utf-8')).hexdigest()
        if md5 != args.md5:
            raise ValueError(f'Expected MD5 {args.md5}, but got {md5}.')
    
    return all_images[args.range_low:args.range_high]


def process_single_image(image_path, args, output_path, tag_and_segment, reconstruct_3d, pose_estimator=None):
    """Process a single image and return results with timing"""
    start_time = time.time()
    
    try:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_path, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Run detection and segmentation
        detections = tag_and_segment(image)
        
        if not detections:
            logging.warning(f"No objects detected in {image_name}")
            return {'image_name': image_name, 'detections': [], 'processing_time': time.time() - start_time}
        
        # Run 3D reconstruction
        detections_with_3d = reconstruct_3d(image, detections)
        
        # Run pose estimation if enabled
        if args.enable_pose3d and pose_estimator:
            detections_with_pose = pose_estimator(image, detections_with_3d)
        else:
            detections_with_pose = detections_with_3d
        
        # Save results
        result = {
            'image_name': image_name,
            'detections': detections_with_pose,
            'processing_time': time.time() - start_time
        }
        
        # Save annotations
        annotations_dir = os.path.join(image_output_dir, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)
        
        with open(os.path.join(annotations_dir, f'{image_name}.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        # Generate summary
        with open(os.path.join(annotations_dir, f'{image_name}_summary.txt'), 'w') as f:
            f.write(f"Image: {image_name}\n")
            f.write(f"Objects Detected: {len(detections_with_pose)}\n")
            f.write(f"Processing Time: {result['processing_time']:.2f}s\n\n")
            
            for i, det in enumerate(detections_with_pose):
                f.write(f"Object {i+1}:\n")
                f.write(f"  Class: {det.get('class_name', 'unknown')}\n")
                f.write(f"  Confidence: {det.get('confidence', 0.0):.3f}\n")
                f.write(f"  Has 3D Pose: {'Yes' if 'pcd_orient_bbox' in det and det['pcd_orient_bbox'] else 'No'}\n")
                f.write("\n")
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to process {image_path}: {e}")
        return {'image_name': os.path.splitext(os.path.basename(image_path))[0], 'detections': [], 'processing_time': time.time() - start_time}


def process_batch(image_paths, args, output_path, tag_and_segment, reconstruct_3d, pose_estimator=None):
    """Process a batch of images with optional parallel processing"""
    results = []
    successful = 0
    failed = 0
    
    if hasattr(args, 'max_workers') and args.max_workers > 1:
        max_workers = min(args.max_workers, len(image_paths), multiprocessing.cpu_count())
        logging.info(f"Using parallel processing with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(process_single_image, path, args, output_path, 
                              tag_and_segment, reconstruct_3d, pose_estimator): path
                for path in image_paths
            }
            
            for future in as_completed(future_to_path):
                result = future.result()
                results.append(result)
                if result['detections']:
                    successful += 1
                else:
                    failed += 1
    else:
        for path in image_paths:
            result = process_single_image(path, args, output_path, tag_and_segment, reconstruct_3d, pose_estimator)
            results.append(result)
            if result['detections']:
                successful += 1
            else:
                failed += 1
    
    logging.info(f"Batch processing complete: {successful} successful, {failed} failed")
    return results


def main():
    """Main function with rolling statistics and checkpointing"""
    args = parse_args()
    setup_logging_with_level(args.log_level)
    
    logging.info("Starting 3D ground truth generation pipeline with ROLLING STATISTICS")
    logging.info(f"Image directory: {args.image_path}")
    logging.info(f"Output directory: {args.output_path}")
    logging.info(f"Stats directory: {args.stats_path}")
    logging.info(f"Checkpoint interval: {args.checkpoint_interval} images")
    
    # Create stats directory
    os.makedirs(args.stats_path, exist_ok=True)
    
    try:
        # Load rolling statistics (resume from checkpoint if requested)
        if args.resume_from_checkpoint:
            rolling_stats = load_checkpoint_stats(args.stats_path)
        else:
            rolling_stats = {
                'total_processed': 0,
                'total_objects_detected': 0,
                'unique_classes': set(),
                'class_counts': {},
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'pose_statistics': {'with_pose': 0, 'without_pose': 0},
                'processing_times': [],
                'last_checkpoint_time': time.time(),
                'start_time': time.time()
            }
        
        # Load image list
        all_images = load_all_images(args)
        logging.info(f"Found {len(all_images)} images to process")
        
        # Initialize models
        logging.info("Initializing models...")
        tag_and_segment = TagAndSegment(cfg, args.device)
        reconstruct_3d = Reconstruct3D(cfg, args.device)
        pose_estimator = None
        if args.enable_pose3d:
            pose_estimator = Pose3DOrientAnything(cfg, args.device)
        logging.info("Models initialized successfully")
        
        # Process images in batches with rolling statistics
        for i in range(0, len(all_images), args.batch_size):
            batch_images = all_images[i:i + args.batch_size]
            batch_paths = [os.path.join(args.image_path, img) for img in batch_images]
            
            batch_num = i // args.batch_size + 1
            total_batches = (len(all_images) + args.batch_size - 1) // args.batch_size
            
            logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} images)")
            
            batch_results = process_batch(batch_paths, args, args.output_path, tag_and_segment, reconstruct_3d, pose_estimator)
            
            # Update rolling statistics
            for result in batch_results:
                update_rolling_stats(rolling_stats, result)
            
            # Save checkpoint if interval reached
            if rolling_stats['total_processed'] % args.checkpoint_interval == 0:
                save_checkpoint_stats(rolling_stats, args.stats_path)
                generate_rolling_summary(rolling_stats, args.stats_path)
                
                logging.info(f"Checkpoint saved: {rolling_stats['total_processed']} images processed")
                logging.info(f"Total objects detected: {rolling_stats['total_objects_detected']}")
                logging.info(f"Unique classes: {len(rolling_stats['unique_classes'])}")
            
            logging.info(f"Batch {batch_num} complete. Running totals - Images: {rolling_stats['total_processed']}, Objects: {rolling_stats['total_objects_detected']}")
        
        # Final checkpoint and summary
        save_checkpoint_stats(rolling_stats, args.stats_path)
        generate_rolling_summary(rolling_stats, args.stats_path)
        
        logging.info("Pipeline completed successfully!")
        logging.info(f"Final totals - Images: {rolling_stats['total_processed']}, Objects: {rolling_stats['total_objects_detected']}, Classes: {len(rolling_stats['unique_classes'])}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        # Save checkpoint even on failure
        if 'rolling_stats' in locals():
            save_checkpoint_stats(rolling_stats, args.stats_path)
            generate_rolling_summary(rolling_stats, args.stats_path)
        import traceback
        logging.debug(f"Error details: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
