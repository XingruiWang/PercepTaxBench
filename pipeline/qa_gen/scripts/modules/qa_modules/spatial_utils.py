import math
from typing import Dict, List, Tuple, Any


class SpatialUtils:
    @staticmethod
    def calculate_3d_distance(center1: List[float], center2: List[float]) -> float:
        if len(center1) != 3 or len(center2) != 3:
            return 0.0
        return math.sqrt(sum((center1[i] - center2[i])**2 for i in range(3)))
    
    @staticmethod
    def calculate_spatial_relationships(detected_objects: List[Dict]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        relationships = {}
        
        for i, obj1 in enumerate(detected_objects):
            for j, obj2 in enumerate(detected_objects):
                if i >= j:
                    continue
                
                obj1_name = obj1.get('labeled_name', obj1['class_name'])
                obj2_name = obj2.get('labeled_name', obj2['class_name'])
                
                # Try to get 3D center data from different possible fields
                center1_3d = obj1.get('pcd_center', obj1.get('3d_data', {}).get('center', []))
                center2_3d = obj2.get('pcd_center', obj2.get('3d_data', {}).get('center', []))
                
                # Parse string format if needed
                if isinstance(center1_3d, str):
                    try:
                        center1_3d = [float(x) for x in center1_3d.strip('[]').split()]
                    except:
                        center1_3d = []
                
                if isinstance(center2_3d, str):
                    try:
                        center2_3d = [float(x) for x in center2_3d.strip('[]').split()]
                    except:
                        center2_3d = []
                
                if len(center1_3d) == 3 and len(center2_3d) == 3:
                    distance = SpatialUtils.calculate_3d_distance(center1_3d, center2_3d)
                    
                    dx = center2_3d[0] - center1_3d[0]
                    dy = center2_3d[1] - center1_3d[1]
                    dz = center2_3d[2] - center1_3d[2]
                    
                    # Get object sizes for better distance classification
                    obj1_size = SpatialUtils._get_object_size(obj1)
                    obj2_size = SpatialUtils._get_object_size(obj2)
                    avg_size = (obj1_size[0] + obj1_size[1] + obj1_size[2] + obj2_size[0] + obj2_size[1] + obj2_size[2]) / 6
                    
                    # Use object size to determine meaningful distance thresholds
                    if distance > max(2.0, avg_size * 3):
                        relative_pos = "far from"
                    elif distance > max(1.0, avg_size * 1.5):
                        relative_pos = "at medium distance from"
                    else:
                        relative_pos = "close to"
                    
                    relationships[(obj1_name, obj2_name)] = {
                        'distance': distance,
                        'relative_position': relative_pos,
                        'dx': dx,
                        'dy': dy,
                        'dz': dz,
                        'object1_size': obj1_size,
                        'object2_size': obj2_size,
                        'avg_object_size': avg_size
                    }
        
        return relationships
    
    @staticmethod
    def determine_left_right(center1: List[float], center2: List[float], obj1_size: List[float] = None, obj2_size: List[float] = None) -> str:
        if len(center1) != 3 or len(center2) != 3:
            return "unknown"
        
        x_diff = center2[0] - center1[0]
        
        # Calculate minimum separation threshold based on object sizes
        if obj1_size and obj2_size and len(obj1_size) >= 1 and len(obj2_size) >= 1:
            min_x_separation = max(obj1_size[0], obj2_size[0]) * 0.3  # 30% of larger object width
            min_x_separation = max(min_x_separation, 0.05)  # At least 5cm
        else:
            min_x_separation = 0.1  # Default fallback
        
        if abs(x_diff) < min_x_separation:
            return "directly aligned with"
        elif x_diff > 0:
            return "right"
        else:
            return "left"
    
    @staticmethod
    def determine_above_below(center1: List[float], center2: List[float], obj1_size: List[float] = None, obj2_size: List[float] = None) -> str:
        if len(center1) != 3 or len(center2) != 3:
            return "unknown"
        
        # FIXED: Use Z-axis for above/below, not Y-axis
        z_diff = center2[2] - center1[2]
        
        # Calculate minimum separation threshold based on object sizes
        if obj1_size and obj2_size and len(obj1_size) >= 3 and len(obj2_size) >= 3:
            min_z_separation = max(obj1_size[2], obj2_size[2]) * 0.3  # 30% of larger object height
            min_z_separation = max(min_z_separation, 0.05)  # At least 5cm
        else:
            min_z_separation = 0.1  # Default fallback
        
        if abs(z_diff) < min_z_separation:
            return "at the same height as"
        elif z_diff > 0:
            return "below"
        else:
            return "above"
    
    @staticmethod
    def determine_front_behind(center1: List[float], center2: List[float], obj1_size: List[float] = None, obj2_size: List[float] = None) -> str:
        if len(center1) != 3 or len(center2) != 3:
            return "unknown"
        
        z_diff = center2[2] - center1[2]
        
        # Calculate minimum separation threshold based on object sizes
        if obj1_size and obj2_size and len(obj1_size) >= 3 and len(obj2_size) >= 3:
            min_z_separation = max(obj1_size[2], obj2_size[2]) * 0.3  # 30% of larger object depth
            min_z_separation = max(min_z_separation, 0.05)  # At least 5cm
        else:
            min_z_separation = 0.1  # Default fallback
        
        if abs(z_diff) < min_z_separation:
            return "at the same depth as"
        elif z_diff < 0:
            return "in front of"
        else:
            return "behind"
    
    @staticmethod
    def calculate_orientation_difference(eulers1: List[float], eulers2: List[float]) -> float:
        if len(eulers1) != 3 or len(eulers2) != 3:
            return 0.0
        
        yaw_diff = abs(eulers1[2] - eulers2[2])
        yaw_diff = min(yaw_diff, 360 - yaw_diff)
        
        return yaw_diff
    
    @staticmethod
    def _get_object_size(obj: Dict) -> List[float]:
        """Extract object size from bounding box data"""
        # Default size if no bounding box data available
        default_size = [0.1, 0.1, 0.1]  # 10cm cube
        
        # Try to get 3D bounding box dimensions
        bbox_3d = obj.get('pcd_orient_bbox', {})
        if isinstance(bbox_3d, dict):
            # Extract dimensions from 3D bounding box
            dimensions = bbox_3d.get('dimensions', [])
            if len(dimensions) == 3:
                return [float(d) for d in dimensions]
        
        # Try to get 2D bounding box and estimate 3D size
        bbox_2d = obj.get('bbox', [])
        if len(bbox_2d) == 4:
            # bbox format: [x1, y1, x2, y2]
            width = abs(bbox_2d[2] - bbox_2d[0])
            height = abs(bbox_2d[3] - bbox_2d[1])
            # Estimate depth as average of width and height (rough approximation)
            depth = (width + height) / 2
            return [width, height, depth]
        
        # Try to get size from other fields
        size_fields = ['size', 'dimensions', 'scale']
        for field in size_fields:
            if field in obj and obj[field]:
                size_data = obj[field]
                if isinstance(size_data, (list, tuple)) and len(size_data) >= 3:
                    return [float(x) for x in size_data[:3]]
        
        return default_size

