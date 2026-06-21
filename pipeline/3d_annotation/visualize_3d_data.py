#!/usr/bin/env python3
"""
3D Data Visualization Module
Generates 3D visualizations from annotation data for spatial reasoning analysis.
Based on actual pipeline outputs and function signatures.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import cv2
import ast

logger = logging.getLogger(__name__)

def visualize_3d_data(annotation_data: Union[str, Dict], image_data: Union[str, np.ndarray], output_path: str) -> Optional[str]:
    """
    Generate 3D visualization from annotation data.
    
    Supports two calling patterns:
    1. visualize_3d_data(annotation_path, image_path, output_dir) -> str
    2. visualize_3d_data(annotation_dict, image_array, output_path) -> str
    
    Args:
        annotation_data: Either path to JSON file or annotation dictionary
        image_data: Either path to image file or image array
        output_path: Output path for visualization
        
    Returns:
        Path to the generated visualization file, or None if failed
    """
    try:
        # Handle different input types
        if isinstance(annotation_data, str):
            # Case 1: annotation_path, image_path, output_dir
            annotation_path = annotation_data
            image_path = image_data
            output_dir = output_path
            
            # Load annotation data
            with open(annotation_path, 'r') as f:
                annotation_dict = json.load(f)
            
            # Load image
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image_array = np.array(image)
            else:
                logger.warning(f"Image file not found: {image_path}")
                return None
                
            # Generate output path
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{image_name}_3d_visualization.png")
            
        else:
            # Case 2: annotation_dict, image_array, output_path
            annotation_dict = annotation_data
            image_array = image_data
            # output_path is already the full path
            
        # Extract detections and 3D data
        detections = annotation_dict.get('detections', [])
        
        if not detections:
            logger.warning("No detections found in annotation data")
            return None
        
        # Create side-by-side visualization (2D + 3D)
        fig = plt.figure(figsize=(20, 10))
        
        # Left panel: Original image with 2D bounding boxes
        ax1 = fig.add_subplot(121)
        ax1.imshow(image_array)
        ax1.set_title('Original Image with 2D Bounding Boxes', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Right panel: 3D visualization
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Set up the 3D plot
        ax2.set_xlabel('X (Left/Right)', fontsize=12)
        ax2.set_ylabel('Y (Depth/Front)', fontsize=12)
        ax2.set_zlabel('Z (Height)', fontsize=12)
        ax2.set_title('3D Scene with Orientation Vectors (X:Left/Right, Y:Depth/Front, Z:Height)', fontsize=14, fontweight='bold')
        
        # Add grid for better depth perception
        ax2.grid(True, alpha=0.3)
        
        # Plot each detected object
        # Use consistent red color for all bounding boxes (like the reference image)
        colors = ['red'] * len(detections)
        
        for i, detection in enumerate(detections):
            try:
                # Extract 3D position from pcd_center
                pcd_center_str = detection.get('pcd_center', '[0, 0, 0]')
                if isinstance(pcd_center_str, str):
                    # Parse string representation of array
                    pcd_center = parse_array_string(pcd_center_str)
                else:
                    pcd_center = pcd_center_str
                
                x, y, z = pcd_center
                
                # Extract 3D bounding box information
                pcd_orient_bbox = detection.get('pcd_orient_bbox', {})
                if pcd_orient_bbox:
                    center_str = pcd_orient_bbox.get('center', '[0, 0, 0]')
                    extent_str = pcd_orient_bbox.get('extent', '[1, 1, 1]')
                    eulers_str = pcd_orient_bbox.get('eulers', '[0, 0, 0]')
                    
                    center = parse_array_string(center_str)
                    extent = parse_array_string(extent_str)
                    eulers = parse_array_string(eulers_str)
                else:
                    # Fallback to simple position
                    center = pcd_center
                    extent = [1, 1, 1]
                    eulers = [0, 0, 0]
                
                # Extract 2D bounding box for both visualizations
                xyxy_str = detection.get('xyxy', '[0, 0, 100, 100]')
                if isinstance(xyxy_str, str):
                    xyxy = parse_array_string(xyxy_str)
                else:
                    xyxy = xyxy_str
                
                # Create object representation
                color = colors[i]
                class_name = detection.get('class_name', f'Object_{i}')
                confidence = detection.get('confidence', 0.0)
                label = f"{class_name} ({confidence:.2f})"
                
                # Plot 2D bounding box on left panel
                x1, y1, x2, y2 = xyxy
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor=color, linewidth=2)
                ax1.add_patch(rect)
                ax1.text(x1, y1-5, label, color=color, fontsize=10, fontweight='bold')
                
                # Plot 3D box on right panel - flip X coordinate to match 2D image orientation
                flipped_center = [-center[0], center[1], center[2]]  # Flip X coordinate
                plot_3d_oriented_box(ax2, flipped_center, extent, eulers, color, label)
                
            except Exception as e:
                logger.warning(f"Failed to plot detection {i}: {e}")
                continue
        
        # Set reasonable axis limits based on data for 3D plot
        all_centers = []
        for detection in detections:
            try:
                pcd_center_str = detection.get('pcd_center', '[0, 0, 0]')
                if isinstance(pcd_center_str, str):
                    center = parse_array_string(pcd_center_str)
                else:
                    center = pcd_center_str
                all_centers.append(center)
            except:
                continue
        
        if all_centers:
            centers_array = np.array(all_centers)
            x_min, x_max = centers_array[:, 0].min() - 2, centers_array[:, 0].max() + 2
            y_min, y_max = centers_array[:, 1].min() - 2, centers_array[:, 1].max() + 2
            z_min, z_max = centers_array[:, 2].min() - 1, centers_array[:, 2].max() + 1
            
            ax2.set_xlim([x_min, x_max])
            ax2.set_ylim([y_min, y_max])
            ax2.set_zlim([z_min, z_max])
        else:
            # Default limits
            ax2.set_xlim([-5, 5])
            ax2.set_ylim([-5, 5])
            ax2.set_zlim([-2, 3])
        
        # Add pose vector legend to 3D plot
        from matplotlib.patches import Patch
        pose_legend_elements = [
            Patch(facecolor='red', label='X-axis (Left/Right)'),
            Patch(facecolor='green', label='Y-axis (Depth/Front)'),
            Patch(facecolor='blue', label='Z-axis (Height)')
        ]
        ax2.legend(handles=pose_legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left', title='Orientation Vectors')
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"3D visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to generate 3D visualization: {e}")
        import traceback
        logger.debug(f"Error details: {traceback.format_exc()}")
        return None

def parse_array_string(array_str: str) -> List[float]:
    """Parse string representation of numpy array to list of floats."""
    try:
        # Remove brackets and whitespace
        clean_str = array_str.strip('[]').strip()
        
        # Try comma-separated first (most common format)
        if ',' in clean_str:
            parts = [part.strip() for part in clean_str.split(',') if part.strip()]
        else:
            # Fallback to space-separated
            parts = [part.strip() for part in clean_str.split() if part.strip()]
        
        return [float(part) for part in parts]
    except Exception as e:
        logger.warning(f"Failed to parse array string '{array_str}': {e}")
        return [0.0, 0.0, 0.0]

def plot_3d_oriented_box(ax, center: List[float], extent: List[float], eulers: List[float], color: Tuple, label: str):
    """Plot a 3D oriented box representing an object with pose vectors."""
    try:
        x, y, z = center
        dx, dy, dz = extent[0] / 2, extent[1] / 2, extent[2] / 2
        
        # Create simple axis-aligned box (no rotation for now to avoid slanted boxes)
        # Create the 8 corners of the box
        corners = np.array([
            [x-dx, y-dy, z-dz],  # 0
            [x+dx, y-dy, z-dz],  # 1
            [x+dx, y+dy, z-dz],  # 2
            [x-dx, y+dy, z-dz],  # 3
            [x-dx, y-dy, z+dz],  # 4
            [x+dx, y-dy, z+dz],  # 5
            [x+dx, y+dy, z+dz],  # 6
            [x-dx, y+dy, z+dz],  # 7
        ])
        
        # Define the edges of the box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        # Plot the edges
        for edge in edges:
            start = corners[edge[0]]
            end = corners[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   color=color, linewidth=2, alpha=0.8)
        
        # Plot yellow cube at center (like in reference image)
        ax.scatter([x], [y], [z], color='yellow', s=100, marker='s', alpha=0.9, edgecolors='black', linewidth=1)
        
        # Get scene bounds to normalize vector size
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        
        # Calculate normalized vector length (1/10th of smallest scene dimension)
        scene_size = min(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
        arrow_length = scene_size * 0.1  # Fixed size relative to scene, not object
        
        # Plot pose vectors (orientation arrows) - X-axis flipped to match 2D image orientation
        ax.quiver(x, y, z, -arrow_length, 0, 0, color='red', arrow_length_ratio=0.15, linewidth=4, alpha=0.9)
        ax.quiver(x, y, z, 0, arrow_length, 0, color='green', arrow_length_ratio=0.15, linewidth=4, alpha=0.9)
        ax.quiver(x, y, z, 0, 0, arrow_length, color='blue', arrow_length_ratio=0.15, linewidth=4, alpha=0.9)
        
        # Add label above the object with better positioning
        ax.text(x, y, z + max(extent) * 0.5, label, fontsize=9, color=color, ha='center', va='bottom', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color))
        
    except Exception as e:
        logger.warning(f"Failed to plot 3D oriented box: {e}")

def create_simple_visualization(annotation_data: Dict, output_path: str) -> bool:
    """
    Create a simple 2D visualization showing object positions.
    
    Args:
        annotation_data: The annotation data dictionary
        output_path: Path to save the visualization
        
    Returns:
        True if successful, False otherwise
    """
    try:
        detections = annotation_data.get('detections', [])
        
        if not detections:
            logger.warning("No detections found for visualization")
            return False
        
        # Create a simple 2D plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each detection
        for i, detection in enumerate(detections):
            try:
                # Extract 2D bounding box
                xyxy_str = detection.get('xyxy', '[0, 0, 100, 100]')
                if isinstance(xyxy_str, str):
                    xyxy = parse_array_string(xyxy_str)
                else:
                    xyxy = xyxy_str
                
                class_name = detection.get('class_name', f'Object_{i}')
                confidence = detection.get('confidence', 0.0)
                
                # Extract center point
                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2
                
                # Plot the object
                ax.scatter(center_x, center_y, s=100, label=f"{class_name} ({confidence:.2f})", alpha=0.7)
                ax.annotate(class_name, (center_x, center_y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
                
            except Exception as e:
                logger.warning(f"Failed to plot detection {i}: {e}")
                continue
        
        ax.set_xlabel('X (width)')
        ax.set_ylabel('Y (height)')
        ax.set_title('Object Detection Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Simple visualization saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create simple visualization: {e}")
        return False

if __name__ == "__main__":
    # Test the visualization function
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 3D visualizations from annotation data')
    parser.add_argument('--annotation', type=str, required=True, help='Path to annotation JSON file')
    parser.add_argument('--image', type=str, required=True, help='Path to original image')
    parser.add_argument('--output', type=str, required=True, help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate visualization
    result = visualize_3d_data(args.annotation, args.image, args.output)
    
    if result:
        print(f"Visualization generated successfully: {result}")
    else:
        print("Failed to generate visualization")
