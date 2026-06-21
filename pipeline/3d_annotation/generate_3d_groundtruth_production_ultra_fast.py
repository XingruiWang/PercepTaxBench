#!/usr/bin/env python3
"""
ULTRA-FAST 3D ground truth generation pipeline with maximum optimizations.
Optimized for speed: larger batches, parallel processing, reduced logging, optimized models.
"""

import os
import sys
import json
import logging
import hashlib
import argparse
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
    parser = argparse.ArgumentParser(description='ULTRA-FAST 3D ground truth generation pipeline.')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to directory containing images')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to output directory for results')
    parser.add_argument('--range_low', type=int, default=None,
                       help='Start index for image range')
    parser.add_argument('--range_high', type=int, default=None,
                       help='End index for image range')
    parser.add_argument('--md5', type=str, default=None,
                       help='MD5 hash for image list validation')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0, cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Number of images to process in each batch (optimized for speed)')
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
    parser.add_argument('--log_level', type=str, default='WARNING',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (WARNING for speed)')
    parser.add_argument('--skip_visualizations', action='store_true',
                       help='Skip 3D visualizations for maximum speed')
    parser.add_argument('--skip_object_crops', action='store_true',
                       help='Skip object crop generation for maximum speed')
    return parser.parse_args()


def setup_logging_with_level(level: str):
    """Setup logging with specified level (optimized for speed)"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Minimal logging setup for speed
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline_ultra_fast.log')
        ]
    )


def load_all_images(args, extensions=['.jpg', '.jpeg', '.png']) -> List[str]:
    """Load and validate image list (optimized)"""
    all_images = [
        x for x in os.listdir(args.image_path)
        if any(x.lower().endswith(ext) for ext in extensions)
    ]
    
    # Sort images for consistent ordering
    all_images.sort()
    
    if args.range_low is None and args.range_high is None:
        return all_images
    
    # Validate MD5 hash if range is specified (optional)
    if args.md5 is not None:
        md5 = hashlib.md5(','.join(all_images).encode('utf-8')).hexdigest()
        if md5 != args.md5:
            raise ValueError(f'Expected MD5 {args.md5}, but got {md5}.')
    
    args.range_low = args.range_low if args.range_low is not None else 0
    args.range_high = args.range_high if args.range_high is not None else len(all_images)
    return all_images[args.range_low:args.range_high]


def create_annotation_summary(annot, output_path):
    """Create a lightweight annotation summary (optimized)"""
    # Extract only the essential information for visualization
    summary = {
        "image_info": {
            "file_path": annot["image_info"]["file_path"],
            "height": annot["image_info"]["height"],
            "width": annot["image_info"]["width"],
            "height_resized": annot["image_info"]["height_resized"],
            "width_resized": annot["image_info"]["width_resized"]
        },
        "detections": []
    }
    
    # Process each detection to extract only what's needed for visualization
    for det in annot["detections"]:
        if "xyxy" in det and "class_name" in det:
            # Create a clean detection entry with just bounding box and label
            clean_det = {
                "bbox": det["xyxy"],  # [x1, y1, x2, y2] in pixel coordinates
                "label": det["numbered_label"] if "numbered_label" in det else det["class_name"],
                "confidence": det.get("confidence", 1.0)
            }
            summary["detections"].append(clean_det)
    
    # Write summary to file
    with open(output_path, 'w') as f:
        f.write("Annotation Summary\n")
        f.write("==================\n\n")
        f.write(f"Image: {summary['image_info']['file_path']}\n")
        f.write(f"Dimensions: {summary['image_info']['width']}x{summary['image_info']['height']}\n")
        f.write(f"Resized: {summary['image_info']['width_resized']}x{summary['image_info']['height_resized']}\n\n")
        
        f.write(f"Detected Objects: {len(summary['detections'])}\n")
        f.write("-" * 50 + "\n")
        
        for i, det in enumerate(summary["detections"], 1):
            f.write(f"{i}. {det['label']}\n")
            f.write(f"   BBox: {det['bbox']}\n")
            f.write(f"   Confidence: {det['confidence']:.3f}\n\n")


def process_single_image_fast(image_path, args, output_path, tag_and_segment, reconstruct_3d, pose_estimator=None):
    """Process a single image through the pipeline with speed optimizations."""
    try:
        # Extract image name and create output directory
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.join(output_path, image_name)
        os.makedirs(image_dir, exist_ok=True)
        
        # Create subdirectories
        annotations_dir = os.path.join(image_dir, "annotations")
        image_visualizations_dir = os.path.join(image_dir, "visualizations")
        object_crops_dir = os.path.join(image_dir, "object_crops")
        os.makedirs(annotations_dir, exist_ok=True)
        os.makedirs(image_visualizations_dir, exist_ok=True)
        os.makedirs(object_crops_dir, exist_ok=True)
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Get image dimensions
        height, width = image_array.shape[:2]
        
        # Process with TagAndSegment (object detection and segmentation)
        logging.debug(f"Processing image: {image_name}")
        detection_results = tag_and_segment.process_image(image_array, image_path)
        
        if not detection_results or 'detections' not in detection_results:
            logging.warning(f"No detections found for {image_name}")
            return None
        
        # Process with Reconstruct3D (3D reconstruction)
        reconstruction_results = reconstruct_3d.process_image(image_array, detection_results)
        
        # Process with Pose3D if enabled
        pose_results = None
        if pose_estimator and args.enable_pose3d:
            pose_results = pose_estimator.process_image(image_array, detection_results)
        
        # Combine all results
        combined_results = {
            "image_info": {
                "file_path": image_path,
                "height": height,
                "width": width,
                "height_resized": height,
                "width_resized": width
            },
            "detections": detection_results.get('detections', []),
            "reconstruction": reconstruction_results,
            "pose": pose_results
        }
        
        # Save annotations
        annotation_path = os.path.join(annotations_dir, f"{image_name}.json")
        with open(annotation_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        # Generate 3D visualization (skip if disabled for speed)
        if not args.skip_visualizations:
            try:
                viz_path = os.path.join(image_visualizations_dir, f"{image_name}_3d_visualization.png")
                visualize_3d_data(combined_results, image_array, viz_path)
            except Exception as e:
                logging.warning(f"Failed to generate 3D visualization for {image_name}: {e}")
        
        # Generate object crops (skip if disabled for speed)
        if not args.skip_object_crops:
            try:
                from generate_cropped_objects import generate_cropped_objects
                generate_cropped_objects(annotation_path, image_path, object_crops_dir)
            except Exception as e:
                logging.warning(f"Failed to generate object crops for {image_name}: {e}")
        
        # Create annotation summary
        summary_path = os.path.join(annotations_dir, f"{image_name}_summary.txt")
        create_annotation_summary(combined_results, summary_path)
        
        logging.debug(f"Successfully processed {image_name}")
        return combined_results
        
    except Exception as e:
        logging.error(f"Failed to process {image_path}: {e}")
        return None


def process_batch_fast(batch_paths, args, output_path, tag_and_segment, reconstruct_3d, pose_estimator):
    """Process a batch of images with parallel processing for speed."""
    results = []
    
    # Use ThreadPoolExecutor for I/O-bound operations
    max_workers = min(args.max_workers, len(batch_paths), multiprocessing.cpu_count())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_single_image_fast, path, args, output_path, 
                          tag_and_segment, reconstruct_3d, pose_estimator): path
            for path in batch_paths
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logging.error(f"Exception in batch processing for {path}: {e}")
    
    return results


def generate_dataset_statistics_fast(all_results, output_path, total_images):
    """Generate dataset statistics (optimized version)"""
    try:
        stats = {
            "dataset_summary": {
                "total_images": total_images,
                "successfully_processed": len(all_results),
                "failed_images": total_images - len(all_results),
                "success_rate": len(all_results) / total_images * 100 if total_images > 0 else 0
            },
            "object_statistics": {
                "total_objects_detected": 0,
                "unique_classes": set(),
                "class_counts": {},
                "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
                "pose_statistics": {"with_pose": 0, "without_pose": 0}
            }
        }
        
        # Process results efficiently
        for result in all_results:
            if 'detections' in result:
                for det in result['detections']:
                    stats["object_statistics"]["total_objects_detected"] += 1
                    
                    class_name = det.get('class_name', 'unknown')
                    stats["object_statistics"]["unique_classes"].add(class_name)
                    stats["object_statistics"]["class_counts"][class_name] = \
                        stats["object_statistics"]["class_counts"].get(class_name, 0) + 1
                    
                    # Confidence distribution
                    confidence = det.get('confidence', 0.5)
                    if confidence >= 0.8:
                        stats["object_statistics"]["confidence_distribution"]["high"] += 1
                    elif confidence >= 0.6:
                        stats["object_statistics"]["confidence_distribution"]["medium"] += 1
                    else:
                        stats["object_statistics"]["confidence_distribution"]["low"] += 1
                    
                    # Pose statistics
                    if result.get('pose') and any(d.get('pose_3d') for d in result.get('pose', {}).get('detections', [])):
                        stats["object_statistics"]["pose_statistics"]["with_pose"] += 1
                    else:
                        stats["object_statistics"]["pose_statistics"]["without_pose"] += 1
        
        # Convert set to list for JSON serialization
        stats["object_statistics"]["unique_classes"] = list(stats["object_statistics"]["unique_classes"])
        
        # Save statistics
        summary_path = os.path.join(output_path, "dataset_statistics.json")
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save human-readable summary
        human_summary_path = os.path.join(output_path, "dataset_summary.txt")
        with open(human_summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ULTRA-FAST PIPELINE PROCESSING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Images: {stats['dataset_summary']['total_images']}\n")
            f.write(f"Successfully Processed: {stats['dataset_summary']['successfully_processed']}\n")
            f.write(f"Failed Images: {stats['dataset_summary']['failed_images']}\n")
            f.write(f"Success Rate: {stats['dataset_summary']['success_rate']:.2f}%\n\n")
            
            f.write(f"Total Objects Detected: {stats['object_statistics']['total_objects_detected']}\n")
            f.write(f"Unique Object Classes: {len(stats['object_statistics']['unique_classes'])}\n\n")
            
            f.write("Top 20 Object Classes by Frequency:\n")
            f.write("-" * 40 + "\n")
            sorted_classes = sorted(stats['object_statistics']['class_counts'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for i, (class_name, count) in enumerate(sorted_classes[:20]):
                f.write(f"{i+1:2d}. {class_name:<20} : {count:>6}\n")
        
        logging.info(f"Dataset statistics saved to: {summary_path}")
        return stats
        
    except Exception as e:
        logging.error(f"Failed to generate dataset statistics: {e}")
        return None


def main():
    """Main function to run the ULTRA-FAST pipeline."""
    args = parse_args()
    setup_logging_with_level(args.log_level)
    
    logging.info("Starting ULTRA-FAST 3D ground truth generation pipeline")
    logging.info(f"Image directory: {args.image_path}")
    logging.info(f"Output directory: {args.output_path}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Batch size: {args.batch_size} (optimized for speed)")
    logging.info(f"Max workers: {args.max_workers}")
    logging.info(f"Skip visualizations: {args.skip_visualizations}")
    logging.info(f"Skip object crops: {args.skip_object_crops}")
    
    try:
        # Load image list
        all_images = load_all_images(args)
        logging.info(f"Found {len(all_images)} images to process")
        
        if len(all_images) == 0:
            logging.error("No images found to process")
            return
        
        # Create output directories
        os.makedirs(args.output_path, exist_ok=True)
        
        # Initialize models once, outside the loop
        logging.info("Initializing models with speed optimizations...")
        tag_and_segment = TagAndSegment(cfg, args.device)
        reconstruct_3d = Reconstruct3D(cfg, args.device)
        pose_estimator = None
        if args.enable_pose3d:
            pose_estimator = Pose3DOrientAnything(cfg, args.device)
        logging.info("Models initialized successfully")
        
        # Process images in optimized batches
        all_results = []
        total_objects_detected = 0
        unique_classes_seen = set()
        
        for i in range(0, len(all_images), args.batch_size):
            batch_images = all_images[i:i + args.batch_size]
            batch_paths = [os.path.join(args.image_path, img) for img in batch_images]
            
            logging.info(f"Processing batch {i//args.batch_size + 1}/{(len(all_images) + args.batch_size - 1)//args.batch_size}")
            logging.info(f"Batch size: {len(batch_paths)}")
            
            batch_results = process_batch_fast(batch_paths, args, args.output_path, 
                                            tag_and_segment, reconstruct_3d, pose_estimator)
            all_results.extend(batch_results)
            
            # Update running statistics
            for result in batch_results:
                if result and 'detections' in result:
                    total_objects_detected += len(result['detections'])
                    for det in result['detections']:
                        unique_classes_seen.add(det.get('class_name', 'unknown'))
            
            logging.info(f"Batch {i//args.batch_size + 1} complete: {len(batch_results)} successful results")
            logging.info(f"Running totals - Images: {len(all_results)}, Objects: {total_objects_detected}, Classes: {len(unique_classes_seen)}")
        
        # Generate dataset-wide statistics
        logging.info(f"All batches complete. Total results collected: {len(all_results)}")
        generate_dataset_statistics_fast(all_results, args.output_path, len(all_images))
        
        logging.info("ULTRA-FAST pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"ULTRA-FAST pipeline failed: {e}")
        import traceback
        logging.debug(f"Error details: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
