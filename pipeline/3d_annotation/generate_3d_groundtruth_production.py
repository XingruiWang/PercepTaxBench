#!/usr/bin/env python3
"""
Generate 3D ground truth using full pipeline with restructured output organization.
Each image gets its own directory containing annotations, visualizations, crops, and QA pairs.
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
    parser = argparse.ArgumentParser(description='Generate 3D ground truth using full pipeline with restructured outputs.')
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
            logging.FileHandler('pipeline.log')
        ]
    )


def is_image_processed(image_name, output_path, require_crops=True, require_visualizations=True):
    """Check if an image has already been processed with all required outputs."""
    image_output_dir = os.path.join(output_path, image_name)
    
    # Basic check: does the output directory exist?
    if not os.path.exists(image_output_dir) or not os.path.isdir(image_output_dir):
        return False
    
    # Check for annotations (required)
    annotations_dir = os.path.join(image_output_dir, 'annotations')
    if not os.path.exists(annotations_dir):
        return False
    
    # Check for object crops if required
    if require_crops:
        crops_dir = os.path.join(image_output_dir, 'object_crops')
        if not os.path.exists(crops_dir):
            return False
        # Check if crops directory has actual crop files (not just empty)
        crop_files = [f for f in os.listdir(crops_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if len(crop_files) == 0:
            return False
    
    # Check for visualizations if required
    if require_visualizations:
        viz_dir = os.path.join(image_output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            return False
        # Check if visualizations directory has actual visualization files
        viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if len(viz_files) == 0:
            return False
    
    return True


def load_all_images(args, extensions=['.jpg', '.jpeg', '.png']) -> List[str]:
    """Load and validate image list, optionally filtering out already processed images"""
    all_images = [
        x for x in os.listdir(args.image_path)
        if any(x.lower().endswith(ext) for ext in extensions)
    ]
    
    # Sort images for consistent ordering
    all_images.sort()
    
    # Filter out already processed images if output directory exists
    if os.path.exists(args.output_path):
        unprocessed_images = []
        for img in all_images:
            image_name = os.path.splitext(img)[0]
            # Check if image is fully processed with crops and visualizations
            if not is_image_processed(image_name, args.output_path, require_crops=True, require_visualizations=True):
                unprocessed_images.append(img)
            else:
                logging.debug(f"Skipping already processed image: {img}")
        
        logging.info(f"Found {len(all_images)} total images, {len(unprocessed_images)} unprocessed")
        all_images = unprocessed_images
    
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
    """Create a lightweight annotation summary with just bounding boxes and labels."""
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


def process_single_image(image_path, args, output_path, tag_and_segment, reconstruct_3d, pose_estimator=None):
    """Process a single image through the full pipeline with restructured outputs."""
    try:
        
        # Extract image name early for use throughout the function
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        logging.info(f"Processing: {os.path.basename(image_path)}")
        
        # Step 1: Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        logging.info(f"Loaded image: {image.size}")
        
        # Step 2: Object detection and segmentation (using pre-initialized models)
        # Create initial annotation structure
        initial_annot = {
            "image_info": {
                "file_path": image_path,
                "height": image.size[1],
                "width": image.size[0],
                "height_resized": image.size[1],
                "width_resized": image.size[0]
            }
        }
        # Get detections from tag and segment
        annot_with_detections = tag_and_segment(image, initial_annot)
        detections = annot_with_detections.get('detections', [])
        logging.info(f"Detected {len(detections)} objects")
        
        # Step 3: 3D reconstruction (using pre-initialized models)
        annot_with_3d = reconstruct_3d(image, annot_with_detections)
        pcd_data = annot_with_3d.get('pcd', {})
        logging.info(f"Generated 3D reconstruction data")
        
        # Step 4: 3D pose estimation (if enabled, using pre-initialized models)
        if args.enable_pose3d and pose_estimator is not None:
            annot_with_pose = pose_estimator(image, annot_with_3d)
            pose_data = annot_with_pose.get('pcd_orient_bbox', {})
            logging.info(f"Generated 3D pose data")
        else:
            pose_data = None
        
        # Step 5: Use the final processed annotation from the pipeline
        if args.enable_pose3d:
            # Use the annotation with pose data
            annot = annot_with_pose
        else:
            # Use the annotation with 3D reconstruction data
            annot = annot_with_3d
        
        # Ensure we have the basic structure
        if 'image_info' not in annot:
            annot['image_info'] = {
                "file_path": image_path,
                "height": image.size[1],
                "width": image.size[0],
                "height_resized": image.size[1],
                "width_resized": image.size[0]
            }
        
        # Add object numbering to JSON annotations for consistency
        # Track category counts for numbering objects of the same class
        category_counts = {}
        for det in annot.get('detections', []):
            if 'class_name' in det:
                label = det['class_name']
                if label not in category_counts:
                    category_counts[label] = 0
                category_counts[label] += 1
                
                # Add numbered label to the detection
                if category_counts[label] > 1:
                    det['numbered_label'] = f"{label} #{category_counts[label]}"
                else:
                    det['numbered_label'] = label
        
        # Skip pose validation for now - it's causing more problems than it solves
        # We'll focus on generating 3D visualizations without validation
        if args.enable_pose3d and 'detections' in annot:
            detections_with_pose = [det for det in annot['detections'] if 'pcd_orient_bbox' in det and 'eulers' in det['pcd_orient_bbox']]
            logging.info(f"Found {len(detections_with_pose)} detections with pose data (validation skipped)")
            # Add a simple note that pose validation was skipped
            annot['pose_validation_summary'] = {'status': 'skipped', 'note': 'Pose validation disabled to improve stability'}
        
        # Store original detections for semantic visualization before pose filtering
        try:
            detections_data = annot.get('detections', [])
            if detections_data is None:
                original_detections = []
                logging.warning(f"No detections found in annotations for {image_name}")
            else:
                original_detections = detections_data.copy()
                logging.info(f"Stored {len(original_detections)} original detections for {image_name}")
        except Exception as e:
            logging.warning(f"Error processing detections for {image_name}: {e}")
            original_detections = []
        
        # Simple filtering - just keep detections with basic requirements (bbox and confidence)
        if args.enable_pose_filtering and 'detections' in annot:
            logging.info(f"Filtering detections: {len(annot['detections'])} before filtering")
            
            # Simple filtering criteria - just need bounding box and reasonable confidence
            confidence_threshold = 0.1  # Very low threshold to keep most detections
            
            filtered_detections = []
            for i, det in enumerate(annot['detections']):
                # Check if detection has required data
                has_bbox = 'xyxy' in det and 'confidence' in det
                confidence_ok = det.get('confidence', 0) >= confidence_threshold
                
                # Apply filtering criteria - be less strict for semantic visualization
                # We need at least bbox and confidence for crops, but 3D data is optional
                basic_requirements_met = has_bbox and confidence_ok
                
                if basic_requirements_met:
                    # For semantic visualization, we can use detections with just 2D data
                    filtered_detections.append(det)
                    logging.info(f"Kept detection {det.get('class_name', 'unknown')} (confidence: {det.get('confidence', 0):.3f}) - bbox exists")
                else:
                    logging.info(f"Filtered out detection {det.get('class_name', 'unknown')} (confidence: {det.get('confidence', 0):.3f}) - missing basic requirements")
            
            annot['detections'] = filtered_detections
            logging.info(f"Filtered detections: {len(annot['detections'])} after filtering")
        
        # Create image-specific output directory
        image_output_dir = os.path.join(args.output_path, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Create subdirectories within the image directory
        image_annotations_dir = os.path.join(image_output_dir, 'annotations')
        os.makedirs(image_annotations_dir, exist_ok=True)
        
        if args.generate_annotations:
            image_visualizations_dir = os.path.join(image_output_dir, 'visualizations')
            os.makedirs(image_visualizations_dir, exist_ok=True)
        
        
        if args.save_pcd:
            image_pcd_dir = os.path.join(image_output_dir, 'pcd')
            os.makedirs(image_pcd_dir, exist_ok=True)
        
        # Create object_crops directory for physical property generation 
        image_crops_dir = os.path.join(image_output_dir, 'object_crops')
        os.makedirs(image_crops_dir, exist_ok=True)
        
        
        # Save annotations
        annotation_filename = os.path.splitext(os.path.basename(image_path))[0] + '.json'
        annotation_path = os.path.join(image_annotations_dir, annotation_filename)
        
        # Save with custom serializer to handle numpy arrays and other non-serializable types
        try:
            logging.info(f"Starting to save annotations for {image_name}")
            with open(annotation_path, 'w') as f:
                json.dump(annot, f, default=srdatagen.utils.serialize, indent=2)
            logging.info(f"Successfully saved annotations to: {annotation_path}")
        except Exception as e:
            logging.error(f"Failed to save annotations for {image_name}: {e}")
            # Try to save a more complete version with detection data
            try:
                # First try with str conversion for numpy types
                with open(annotation_path, 'w') as f:
                    json.dump(annot, f, default=str, indent=2)
                logging.info(f"Saved annotations with str conversion to: {annotation_path}")
            except Exception as e2:
                logging.error(f"Failed to save with str conversion: {e2}")
                # Fall back to simplified version but include detection data
                simplified_annot = {
                    'image_info': annot.get('image_info', {}),
                    'detection_count': len(annot.get('detections', [])),
                    'has_detections': len(annot.get('detections', [])) > 0,
                    'detections': annot.get('detections', [])  # Include actual detection data
                }
                with open(annotation_path, 'w') as f:
                    json.dump(simplified_annot, f, indent=2)
                logging.info(f"Saved simplified annotations to: {annotation_path}")
        
        # Create annotation summary
        try:
            logging.info(f"Starting to create annotation summary for {image_name}")
            summary_filename = os.path.splitext(os.path.basename(image_path))[0] + '_summary.txt'
            summary_path = os.path.join(image_annotations_dir, summary_filename)
            create_annotation_summary(annot, summary_path)
            logging.info(f"Successfully saved annotation summary to: {summary_path}")
        except Exception as e:
            logging.error(f"Failed to create annotation summary for {image_name}: {e}")
            # Create a simple summary
            with open(summary_path, 'w') as f:
                f.write(f"Image: {image_name}\n")
                f.write(f"Detections: {len(annot.get('detections', []))}\n")
                f.write(f"Error creating detailed summary: {e}\n")
            logging.info(f"Saved simple annotation summary to: {summary_path}")
        
        # Generate 3D visualizations if pose estimation was enabled (skip if disabled for speed)
        if not args.skip_visualizations and args.enable_pose3d and 'detections' in annot and len(annot['detections']) > 0:
            try:
                # Save annotations to a temporary JSON file for visualization
                temp_annot_path = os.path.join(image_visualizations_dir, f"{image_name}_temp.json")
                with open(temp_annot_path, 'w') as f:
                    json.dump(annot, f, default=str)
                
                # Generate 3D visualization directly in the visualizations directory
                viz_output_path = visualize_3d_data(temp_annot_path, image_path, image_visualizations_dir)
                
                # Clean up temp file
                if os.path.exists(temp_annot_path):
                    os.remove(temp_annot_path)
                
                # Log the visualization generation
                if viz_output_path and os.path.exists(viz_output_path):
                    logging.debug(f"Generated 3D visualization: {viz_output_path}")
                else:
                    logging.warning(f"3D visualization file not found at expected path: {viz_output_path}")
                
            except Exception as e:
                logging.warning(f"Failed to generate 3D visualization: {e}")
                import traceback
                logging.debug(f"3D visualization error details: {traceback.format_exc()}")
        
        # Generate object crops for physical property generation (skip if disabled for speed)
        if not args.skip_object_crops and 'detections' in annot and len(annot['detections']) > 0:
            try:
                logging.debug(f"Generating object crops for {image_name}")
                
                # Import the standalone cropping functionality
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                # Note: generate_cropped_objects function needs to be implemented or imported from appropriate location
                # from openimages_3d_annotations.scripts.generate_cropped_objects import generate_cropped_objects
                
                # TODO: Implement object cropping functionality
                # Generate cropped objects
                # crops_success = generate_cropped_objects(
                #     annotation_path,  # Use the saved annotation file
                #     image_path,        # Original image path
                #     image_crops_dir    # Output directory for crops
                # )
                
                # if crops_success:
                #     logging.debug(f"Successfully generated object crops for {image_name}")
                # else:
                #     logging.warning(f"Failed to generate object crops for {image_name}")
                logging.info(f"Object cropping functionality not yet implemented for {image_name}")
                    
            except Exception as e:
                logging.warning(f"Failed to generate object crops for {image_name}: {e}")
                import traceback
                logging.debug(f"Object cropping error details: {traceback.format_exc()}")
        
        return annot
        
    except Exception as e:
        logging.error(f"Failed to process {image_path}: {e}")
        import traceback
        logging.debug(f"Error details: {traceback.format_exc()}")
        return None


def process_batch(image_paths, args, output_path, tag_and_segment, reconstruct_3d, pose_estimator=None):
    """Process a batch of images with optional parallel processing for speed."""
    results = []
    successful = 0
    failed = 0
    
    # Use parallel processing if max_workers > 1
    if hasattr(args, 'max_workers') and args.max_workers > 1:
        max_workers = min(args.max_workers, len(image_paths), multiprocessing.cpu_count())
        logging.info(f"Using parallel processing with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(process_single_image, path, args, output_path, 
                              tag_and_segment, reconstruct_3d, pose_estimator): path
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        successful += 1
                        logging.debug(f"Successfully processed: {os.path.basename(path)}")
                    else:
                        failed += 1
                        logging.warning(f"Failed to process: {os.path.basename(path)}")
                except Exception as e:
                    failed += 1
                    logging.error(f"Error processing {path}: {e}")
                    import traceback
                    logging.debug(f"Error details: {traceback.format_exc()}")
    else:
        # Sequential processing (original method)
        for i, image_path in enumerate(image_paths):
            try:
                logging.debug(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                result = process_single_image(image_path, args, output_path, tag_and_segment, reconstruct_3d, pose_estimator)
                if result is not None:
                    results.append(result)
                    successful += 1
                    logging.debug(f"Successfully processed: {os.path.basename(image_path)}")
                else:
                    failed += 1
                    logging.warning(f"Failed to process: {os.path.basename(image_path)}")
                    
            except Exception as e:
                failed += 1
                logging.error(f"Error processing {image_path}: {e}")
                import traceback
                logging.debug(f"Error details: {traceback.format_exc()}")
    
    logging.info(f"Batch processing complete: {successful} successful, {failed} failed")
    return results


def generate_dataset_statistics(results, output_path, all_images_count):
    """Generate comprehensive statistics for the entire dataset."""
    logging.info("Generating dataset-wide statistics...")
    
    try:
        # Initialize statistics
        stats = {
            'dataset_summary': {
                'total_images': all_images_count,
                'successfully_processed': len(results),
                'failed_images': all_images_count - len(results),
                'success_rate': f"{(len(results) / all_images_count * 100):.2f}%"
            },
            'object_statistics': {
                'total_objects_detected': 0,
                'unique_classes': set(),
                'class_counts': {},
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'pose_statistics': {'with_pose': 0, 'without_pose': 0}
            },
            'detection_details': [],
            'processing_times': []
        }
        
        # Process each result to gather statistics
        for result in results:
            if result and 'detections' in result:
                detections = result['detections']
                stats['object_statistics']['total_objects_detected'] += len(detections)
                
                for det in detections:
                    # Class statistics
                    class_name = det.get('class_name', 'unknown')
                    stats['object_statistics']['unique_classes'].add(class_name)
                    stats['object_statistics']['class_counts'][class_name] = stats['object_statistics']['class_counts'].get(class_name, 0) + 1
                    
                    # Confidence distribution
                    confidence = det.get('confidence', 0.0)
                    if confidence >= 0.8:
                        stats['object_statistics']['confidence_distribution']['high'] += 1
                    elif confidence >= 0.6:
                        stats['object_statistics']['confidence_distribution']['medium'] += 1
                    else:
                        stats['object_statistics']['confidence_distribution']['low'] += 1
                    
                    # Pose statistics
                    if 'pcd_orient_bbox' in det and det['pcd_orient_bbox']:
                        stats['object_statistics']['pose_statistics']['with_pose'] += 1
                    else:
                        stats['object_statistics']['pose_statistics']['without_pose'] += 1
                    
                    # Store detection details for analysis
                    det_detail = {
                        'image': result.get('image_info', {}).get('file_path', 'unknown'),
                        'class': class_name,
                        'confidence': confidence,
                        'has_pose': 'pcd_orient_bbox' in det and det['pcd_orient_bbox'] != {},
                        'bbox_area': det.get('bbox_area', 0) if 'bbox_area' in det else 0
                    }
                    stats['detection_details'].append(det_detail)
        
        # Convert set to list for JSON serialization
        stats['object_statistics']['unique_classes'] = list(stats['object_statistics']['unique_classes'])
        
        # Sort class counts by frequency
        stats['object_statistics']['class_counts'] = dict(sorted(
            stats['object_statistics']['class_counts'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        # Generate summary report
        summary_path = os.path.join(output_path, 'dataset_statistics.json')
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Generate human-readable summary
        human_summary_path = os.path.join(output_path, 'dataset_summary.txt')
        with open(human_summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATASET PROCESSING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Images: {stats['dataset_summary']['total_images']}\n")
            f.write(f"Successfully Processed: {stats['dataset_summary']['successfully_processed']}\n")
            f.write(f"Failed Images: {stats['dataset_summary']['failed_images']}\n")
            f.write(f"Success Rate: {stats['dataset_summary']['success_rate']}\n\n")
            
            f.write(f"Total Objects Detected: {stats['object_statistics']['total_objects_detected']}\n")
            f.write(f"Unique Object Classes: {len(stats['object_statistics']['unique_classes'])}\n\n")
            
            f.write("Top 20 Object Classes by Frequency:\n")
            f.write("-" * 40 + "\n")
            for i, (class_name, count) in enumerate(list(stats['object_statistics']['class_counts'].items())[:20]):
                f.write(f"{i+1:2d}. {class_name:<20} : {count:>6}\n")
            
            f.write("\nConfidence Distribution:\n")
            f.write("-" * 25 + "\n")
            f.write(f"High (≥0.8): {stats['object_statistics']['confidence_distribution']['high']}\n")
            f.write(f"Medium (0.6-0.8): {stats['object_statistics']['confidence_distribution']['medium']}\n")
            f.write(f"Low (<0.6): {stats['object_statistics']['confidence_distribution']['low']}\n\n")
            
            f.write("Pose Estimation Statistics:\n")
            f.write("-" * 28 + "\n")
            f.write(f"Objects with 3D Pose: {stats['object_statistics']['pose_statistics']['with_pose']}\n")
            f.write(f"Objects without 3D Pose: {stats['object_statistics']['pose_statistics']['without_pose']}\n")
            
            if stats['object_statistics']['total_objects_detected'] > 0:
                pose_rate = (stats['object_statistics']['pose_statistics']['with_pose'] / 
                           stats['object_statistics']['total_objects_detected'] * 100)
                f.write(f"Pose Success Rate: {pose_rate:.2f}%\n")
        
        logging.info(f"Dataset statistics saved to: {summary_path}")
        logging.info(f"Human-readable summary saved to: {human_summary_path}")
        
        return stats
        
    except Exception as e:
        logging.error(f"Failed to generate dataset statistics: {e}")
        import traceback
        logging.debug(f"Statistics generation error details: {traceback.format_exc()}")
        return None


def main():
    """Main function to run the pipeline."""
    args = parse_args()
    setup_logging_with_level(args.log_level)
    
    logging.info("Starting 3D ground truth generation pipeline with SPEED OPTIMIZATIONS")
    logging.info(f"Image directory: {args.image_path}")
    logging.info(f"Output directory: {args.output_path}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Batch size: {args.batch_size} (optimized for speed)")
    logging.info(f"Max workers: {getattr(args, 'max_workers', 1)}")
    logging.info(f"Skip visualizations: {getattr(args, 'skip_visualizations', False)}")
    logging.info(f"Skip object crops: {getattr(args, 'skip_object_crops', False)}")
    logging.info(f"Enable 3D pose: {args.enable_pose3d}")
    logging.info(f"Enable pose filtering: {args.enable_pose_filtering}")
    
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
        logging.info("Initializing models...")
        tag_and_segment = TagAndSegment(cfg, args.device)
        reconstruct_3d = Reconstruct3D(cfg, args.device)
        pose_estimator = None
        if args.enable_pose3d:
            pose_estimator = Pose3DOrientAnything(cfg, args.device)
        logging.info("Models initialized successfully")
        
        # Process images in batches
        all_results = []  # Collect all results from all batches
        total_objects_detected = 0
        unique_classes_seen = set()
        
        for i in range(0, len(all_images), args.batch_size):
            batch_images = all_images[i:i + args.batch_size]
            batch_paths = [os.path.join(args.image_path, img) for img in batch_images]
            
            logging.info(f"Processing batch {i//args.batch_size + 1}/{(len(all_images) + args.batch_size - 1)//args.batch_size}")
            logging.info(f"Batch size: {len(batch_paths)}")
            
            batch_results = process_batch(batch_paths, args, args.output_path, tag_and_segment, reconstruct_3d, pose_estimator)
            all_results.extend(batch_results)  # Add batch results to overall collection
            
            # Update running statistics
            for result in batch_results:
                if result and 'detections' in result:
                    total_objects_detected += len(result['detections'])
                    for det in result['detections']:
                        unique_classes_seen.add(det.get('class_name', 'unknown'))
            
            logging.info(f"Batch {i//args.batch_size + 1} complete: {len(batch_results)} successful results")
            logging.info(f"Running totals - Images: {len(all_results)}, Objects: {total_objects_detected}, Classes: {len(unique_classes_seen)}")
        
        # Generate dataset-wide statistics after all images are processed
        logging.info(f"All batches complete. Total results collected: {len(all_results)}")
        logging.info(f"Final totals - Images: {len(all_results)}, Objects: {total_objects_detected}, Classes: {len(unique_classes_seen)}")
        generate_dataset_statistics(all_results, args.output_path, len(all_images))

        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        logging.debug(f"Error details: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
