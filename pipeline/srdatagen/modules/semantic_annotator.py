import cv2
import numpy as np
from PIL import Image
import torch
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import json
import os


class ColorDetector:
    """Extract dominant colors from object regions"""
    
    def __init__(self):
        # Define color ranges in HSV space
        self.color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            'orange': [(10, 50, 50), (25, 255, 255)],
            'yellow': [(25, 50, 50), (35, 255, 255)],
            'green': [(35, 50, 50), (85, 255, 255)],
            'blue': [(85, 50, 50), (130, 255, 255)],
            'purple': [(130, 50, 50), (170, 255, 255)],
            'pink': [(145, 50, 50), (165, 255, 255)],
            'brown': [(10, 50, 50), (20, 255, 255), (0, 50, 20), (20, 255, 200)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'gray': [(0, 0, 100), (180, 30, 200)],
            'black': [(0, 0, 0), (180, 255, 30)]
        }
    
    def extract_dominant_colors(self, image: np.ndarray, bbox: List[int], mask: Optional[np.ndarray] = None) -> Dict:
        """Extract dominant colors from a bounding box region"""
        x1, y1, x2, y2 = bbox
        
        # Crop the object region
        if mask is not None:
            # Use mask to get only the object pixels
            obj_region = image[y1:y2, x1:x2]
            mask_region = mask[y1:y2, x1:x2]
            obj_region = obj_region * mask_region[:, :, np.newaxis]
        else:
            obj_region = image[y1:y2, x1:x2]
        
        if obj_region.size == 0:
            return {'primary': 'unknown', 'secondary': 'unknown', 'confidence': 0.0}
        
        # Convert to HSV
        hsv = cv2.cvtColor(obj_region, cv2.COLOR_RGB2HSV)
        
        # Reshape for clustering
        pixels = hsv.reshape(-1, 3)
        
        # Filter out black/white pixels
        valid_pixels = pixels[(pixels[:, 1] > 30) & (pixels[:, 2] > 30) & (pixels[:, 2] < 225)]
        
        if len(valid_pixels) < 10:
            return {'primary': 'unknown', 'secondary': 'unknown', 'confidence': 0.0}
        
        # Use k-means to find dominant colors
        n_clusters = min(3, len(valid_pixels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(valid_pixels)
        
        # Get cluster centers and sizes
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels)
        
        # Sort by cluster size
        sorted_indices = np.argsort(cluster_sizes)[::-1]
        
        # Map to color names
        primary_color = self._hsv_to_color_name(centers[sorted_indices[0]])
        secondary_color = self._hsv_to_color_name(centers[sorted_indices[1]]) if len(sorted_indices) > 1 else 'unknown'
        
        # Calculate confidence based on cluster separation
        confidence = min(1.0, cluster_sizes[sorted_indices[0]] / len(valid_pixels))
        
        return {
            'primary': primary_color,
            'secondary': secondary_color,
            'confidence': confidence,
            'hsv_values': centers[sorted_indices[0]].tolist()
        }
    
    def _hsv_to_color_name(self, hsv: np.ndarray) -> str:
        """Convert HSV values to color name"""
        h, s, v = hsv
        
        # Check each color range
        for color_name, ranges in self.color_ranges.items():
            if len(ranges) == 2:
                # Single range
                h_low, s_low, v_low = ranges[0]
                h_high, s_high, v_high = ranges[1]
                
                if (h_low <= h <= h_high) and (s_low <= s <= s_high) and (v_low <= v <= v_high):
                    return color_name
            elif len(ranges) == 4:
                # Two ranges (for red which wraps around)
                h_low1, s_low1, v_low1 = ranges[0]
                h_high1, s_high1, v_high1 = ranges[1]
                h_low2, s_low2, v_low2 = ranges[2]
                h_high2, s_high2, v_high2 = ranges[3]
                
                if (((h_low1 <= h <= h_high1) and (s_low1 <= s <= s_high1) and (v_low1 <= v <= v_high1)) or
                    ((h_low2 <= h <= h_high2) and (s_low2 <= s <= s_high2) and (v_low2 <= v <= v_high2))):
                    return color_name
        
        return 'unknown'


class ShapeAnalyzer:
    """Analyze object shapes and geometric properties"""
    
    def __init__(self):
        self.shape_thresholds = {
            'circularity_threshold': 0.7,
            'elongated_threshold': 3.0,
            'square_threshold': 0.8
        }
    
    def analyze_shape(self, bbox: List[int], mask: Optional[np.ndarray] = None) -> Dict:
        """Analyze the shape of an object"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Basic shape properties
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Shape classification based on aspect ratio
        if aspect_ratio > self.shape_thresholds['elongated_threshold']:
            shape_type = 'elongated'
        elif aspect_ratio < 1.0 / self.shape_thresholds['elongated_threshold']:
            shape_type = 'elongated'
        elif abs(aspect_ratio - 1.0) < 0.2:
            shape_type = 'square'
        else:
            shape_type = 'rectangular'
        
        # Size classification
        area = width * height
        if area < 1000:
            size_category = 'small'
        elif area < 10000:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        # If we have a mask, do more detailed analysis
        if mask is not None:
            mask_region = mask[y1:y2, x1:x2]
            detailed_shape = self._analyze_mask_shape(mask_region)
            shape_type = detailed_shape.get('type', shape_type)
        
        return {
            'type': shape_type,
            'aspect_ratio': aspect_ratio,
            'size_category': size_category,
            'width': width,
            'height': height,
            'area': area
        }
    
    def _analyze_mask_shape(self, mask: np.ndarray) -> Dict:
        """Analyze shape using mask for more accurate classification"""
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'type': 'unknown'}
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        if area < 10:
            return {'type': 'unknown'}
        
        # Calculate shape properties
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Fit bounding rectangle
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        rect_aspect = max(rect_width, rect_height) / min(rect_width, rect_height) if min(rect_width, rect_height) > 0 else 1
        
        # Shape classification
        if circularity > self.shape_thresholds['circularity_threshold']:
            shape_type = 'circular'
        elif rect_aspect > self.shape_thresholds['elongated_threshold']:
            shape_type = 'elongated'
        elif abs(rect_aspect - 1.0) < 0.2:
            shape_type = 'square'
        else:
            shape_type = 'rectangular'
        
        return {
            'type': shape_type,
            'circularity': circularity,
            'rect_aspect_ratio': rect_aspect,
            'contour_area': area,
            'perimeter': perimeter
        }


class SpatialRelationDetector:
    """Detect spatial relationships between objects"""
    
    def __init__(self):
        self.relation_thresholds = {
            'near_distance': 100,  # pixels
            'overlap_threshold': 0.3,  # IoU threshold
            'above_threshold': 0.1  # relative height difference
        }
    
    def detect_relations(self, objects: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """Detect spatial relations between all objects"""
        relations = []
        img_height, img_width = image_shape
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                relation = self._analyze_pair_relation(obj1, obj2, img_width, img_height)
                if relation:
                    relations.append({
                        'subject_id': f"obj_{i:03d}",
                        'object_id': f"obj_{j:03d}",
                        'relation_type': relation['type'],
                        'confidence': relation['confidence'],
                        'distance': relation['distance'],
                        'spatial_info': relation['spatial_info']
                    })
        
        return relations
    
    def _analyze_pair_relation(self, obj1: Dict, obj2: Dict, img_width: int, img_height: int) -> Optional[Dict]:
        """Analyze spatial relation between two objects"""
        # Get bounding boxes
        bbox1 = obj1.get('bbox', obj1.get('xyxy', []))
        bbox2 = obj2.get('bbox', obj2.get('xyxy', []))
        
        if len(bbox1) != 4 or len(bbox2) != 4:
            return None
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate centers
        center1_x = (x1_1 + x2_1) / 2
        center1_y = (y1_1 + y2_1) / 2
        center2_x = (x1_2 + x2_2) / 2
        center2_y = (y1_2 + y2_2) / 2
        
        # Calculate distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # Check for overlap
        overlap = self._calculate_overlap(bbox1, bbox2)
        
        # Determine spatial relation
        relation_type = self._classify_spatial_relation(
            bbox1, bbox2, center1_x, center1_y, center2_x, center2_y, 
            distance, overlap, img_width, img_height
        )
        
        if relation_type:
            return {
                'type': relation_type,
                'confidence': self._calculate_relation_confidence(distance, overlap),
                'distance': distance,
                'spatial_info': {
                    'overlap': overlap,
                    'center_distance': distance,
                    'relative_positions': {
                        'obj1_center': [center1_x, center1_y],
                        'obj2_center': [center2_x, center2_y]
                    }
                }
            }
        
        return None
    
    def _calculate_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _classify_spatial_relation(self, bbox1: List[int], bbox2: List[int], 
                                 cx1: float, cy1: float, cx2: float, cy2: float,
                                 distance: float, overlap: float, img_width: int, img_height: int) -> Optional[str]:
        """Classify the spatial relation between two objects"""
        # Check for overlap
        if overlap > self.relation_thresholds['overlap_threshold']:
            return 'overlapping'
        
        # Check for containment
        if self._is_contained(bbox1, bbox2):
            return 'contains'
        if self._is_contained(bbox2, bbox1):
            return 'contained_by'
        
        # Check for adjacency
        if distance < self.relation_thresholds['near_distance']:
            # Determine relative position
            dx = cx2 - cx1
            dy = cy2 - cy1
            
            # Normalize by image dimensions
            dx_norm = dx / img_width
            dy_norm = dy / img_height
            
            # Classify relative position
            if abs(dx_norm) > abs(dy_norm):
                if dx_norm > 0.1:
                    return 'to_right_of'
                elif dx_norm < -0.1:
                    return 'to_left_of'
            else:
                if dy_norm > 0.1:
                    return 'below'
                elif dy_norm < -0.1:
                    return 'above'
            
            return 'near'
        
        return None
    
    def _is_contained(self, bbox1: List[int], bbox2: List[int]) -> bool:
        """Check if bbox1 is contained within bbox2"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        return (x1_2 <= x1_1 and y1_2 <= y1_1 and 
                x2_2 >= x2_1 and y2_2 >= y2_1)
    
    def _calculate_relation_confidence(self, distance: float, overlap: float) -> float:
        """Calculate confidence score for spatial relation"""
        # Higher confidence for closer objects or overlapping objects
        if overlap > 0:
            return min(1.0, overlap + 0.5)
        
        # Distance-based confidence
        if distance < 50:
            return 0.9
        elif distance < 100:
            return 0.7
        elif distance < 200:
            return 0.5
        else:
            return 0.3


class SemanticAnnotator:
    """Main class for semantic annotation of objects"""
    
    def __init__(self):
        self.color_detector = ColorDetector()
        self.shape_analyzer = ShapeAnalyzer()
        self.relation_detector = SpatialRelationDetector()
    
    def annotate_objects(self, image: np.ndarray, detections: List[Dict], 
                        masks: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """Annotate all detected objects with semantic information"""
        annotated_objects = []
        
        for i, detection in enumerate(detections):
            # Get bounding box
            bbox = detection.get('bbox', detection.get('xyxy', []))
            if len(bbox) != 4:
                continue
            
            # Get mask if available
            mask = masks[i] if masks and i < len(masks) else None
            
            # Extract attributes
            attributes = self._extract_attributes(image, bbox, mask, detection)
            
            # Create annotated object
            annotated_obj = {
                'id': f"obj_{i:03d}",
                'detection': {
                    'bbox': bbox,
                    'confidence': detection.get('confidence', 0.0),
                    'class_name': detection.get('class_name', 'unknown')
                },
                'attributes': attributes
            }
            
            annotated_objects.append(annotated_obj)
        
        # Detect spatial relations
        if len(annotated_objects) > 1:
            relations = self.relation_detector.detect_relations(
                annotated_objects, image.shape[:2]
            )
            
            # Add relations to objects
            for relation in relations:
                subject_id = relation['subject_id']
                object_id = relation['object_id']
                
                # Find the objects and add relations
                for obj in annotated_objects:
                    if obj['id'] == subject_id:
                        if 'relations' not in obj:
                            obj['relations'] = []
                        obj['relations'].append(relation)
        
        return annotated_objects
    
    def _extract_attributes(self, image: np.ndarray, bbox: List[int], 
                           mask: Optional[np.ndarray], detection: Dict) -> Dict:
        """Extract all attributes for a single object"""
        # Extract colors
        colors = self.color_detector.extract_dominant_colors(image, bbox, mask)
        
        # Analyze shape
        shape_info = self.shape_analyzer.analyze_shape(bbox, mask)
        
        # Get 3D information if available
        pcd_info = detection.get('pcd', {})
        size_3d = pcd_info.get('size', [0, 0, 0]) if pcd_info else [0, 0, 0]
        
        # Classify size based on 3D dimensions if available
        if any(size > 0 for size in size_3d):
            max_dim = max(size_3d)
            if max_dim < 0.5:
                size_category = 'small'
            elif max_dim < 1.0:
                size_category = 'medium'
            else:
                size_category = 'large'
        else:
            size_category = shape_info['size_category']
        
        return {
            'physical': {
                'color': colors['primary'],
                'secondary_color': colors['secondary'],
                'color_confidence': colors['confidence'],
                'shape': shape_info['type'],
                'size_category': size_category,
                'dimensions_2d': {
                    'width': shape_info['width'],
                    'height': shape_info['height'],
                    'area': shape_info['area']
                }
            },
            'quantitative': {
                'count': 1,
                'bbox_area': shape_info['area'],
                'aspect_ratio': shape_info['aspect_ratio']
            }
        }
    
    def save_annotations(self, annotations: List[Dict], output_path: str):
        """Save semantic annotations to JSON file"""
        output_data = {
            'semantic_annotations': annotations,
            'metadata': {
                'total_objects': len(annotations),
                'annotation_timestamp': str(np.datetime64('now')),
                'version': '1.0'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved semantic annotations to {output_path}")
