import numpy as np
import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class PoseValidationResult:
    """Result of pose validation"""
    is_valid: bool
    confidence_score: float
    validation_checks: Dict[str, bool]
    warnings: List[str]
    errors: List[str]

class PoseValidator:
    """Validates 3D pose predictions for mathematical consistency and physical plausibility"""
    
    def __init__(self):
        # Object-specific pose constraints
        self.object_constraints = {
            # Vehicles - should be mostly upright, limited roll
            'car': {'max_roll': np.pi/6, 'max_elev': np.pi/4, 'min_confidence': 0.6},
            'truck': {'max_roll': np.pi/6, 'max_elev': np.pi/4, 'min_confidence': 0.6},
            'bus': {'max_roll': np.pi/6, 'max_elev': np.pi/4, 'min_confidence': 0.6},
            'motorcycle': {'max_roll': np.pi/4, 'max_elev': np.pi/3, 'min_confidence': 0.6},
            'bicycle': {'max_roll': np.pi/4, 'max_elev': np.pi/3, 'min_confidence': 0.6},
            
            # People - should be upright, limited roll
            'person': {'max_roll': np.pi/6, 'max_elev': np.pi/6, 'min_confidence': 0.7},
            'man': {'max_roll': np.pi/6, 'max_elev': np.pi/6, 'min_confidence': 0.7},
            'woman': {'max_roll': np.pi/6, 'max_elev': np.pi/6, 'min_confidence': 0.7},
            'child': {'max_roll': np.pi/6, 'max_elev': np.pi/6, 'min_confidence': 0.7},
            'boy': {'max_roll': np.pi/6, 'max_elev': np.pi/6, 'min_confidence': 0.7},
            'girl': {'max_roll': np.pi/6, 'max_elev': np.pi/6, 'min_confidence': 0.7},
            
            # Furniture - should be mostly level
            'chair': {'max_roll': np.pi/8, 'max_elev': np.pi/8, 'min_confidence': 0.6},
            'table': {'max_roll': np.pi/8, 'max_elev': np.pi/8, 'min_confidence': 0.6},
            'sofa': {'max_roll': np.pi/8, 'max_elev': np.pi/8, 'min_confidence': 0.6},
            'bed': {'max_roll': np.pi/8, 'max_elev': np.pi/8, 'min_confidence': 0.6},
            
            # Animals - more flexible but reasonable limits
            'dog': {'max_roll': np.pi/3, 'max_elev': np.pi/3, 'min_confidence': 0.6},
            'cat': {'max_roll': np.pi/3, 'max_elev': np.pi/3, 'min_confidence': 0.6},
            'horse': {'max_roll': np.pi/4, 'max_elev': np.pi/4, 'min_confidence': 0.6},
            'bird': {'max_roll': np.pi/2, 'max_elev': np.pi/2, 'min_confidence': 0.6},
            
            # Default constraints for unknown objects
            'default': {'max_roll': np.pi/3, 'max_elev': np.pi/3, 'min_confidence': 0.5}
        }
    
    def validate_pose(self, detection: Dict[str, Any], confidence: float = None) -> PoseValidationResult:
        """Validate a single object's pose"""
        validation_checks = {}
        warnings = []
        errors = []
        
        # Get pose data
        pose_data = detection.get('pcd_orient_bbox', {})
        eulers = pose_data.get('eulers', [0.0, 0.0, 0.0])
        class_name = detection.get('class_name', 'unknown')
        
        if not eulers or len(eulers) != 3:
            errors.append("Missing or invalid Euler angles")
            return PoseValidationResult(False, 0.0, {}, warnings, errors)
        
        roll, pitch, yaw = eulers
        
        # 1. Check mathematical consistency
        validation_checks['euler_range'] = self._check_euler_range(roll, pitch, yaw)
        if not validation_checks['euler_range']:
            errors.append("Euler angles outside valid range (-π to π)")
        
        # 2. Check physical plausibility
        validation_checks['physical_plausibility'] = self._check_physical_plausibility(
            roll, pitch, yaw, class_name
        )
        if not validation_checks['physical_plausibility']:
            warnings.append(f"Pose angles may be physically implausible for {class_name}")
        
        # 3. Check confidence threshold
        min_confidence = self._get_min_confidence(class_name)
        validation_checks['confidence_threshold'] = confidence >= min_confidence if confidence else True
        if confidence and not validation_checks['confidence_threshold']:
            warnings.append(f"Confidence {confidence:.3f} below threshold {min_confidence:.3f} for {class_name}")
        
        # 4. Check for extreme angles
        validation_checks['extreme_angles'] = self._check_extreme_angles(roll, pitch, yaw)
        if not validation_checks['extreme_angles']:
            warnings.append("One or more angles are extremely large")
        
        # 5. Check for zero/identity poses
        validation_checks['non_identity'] = self._check_non_identity_pose(roll, pitch, yaw)
        if not validation_checks['non_identity']:
            warnings.append("Pose appears to be identity (all angles near zero)")
        
        # Overall validation result - ensure all values are boolean
        is_valid = all(bool(v) for v in validation_checks.values())
        
        return PoseValidationResult(
            is_valid=is_valid,
            confidence_score=confidence or 0.0,
            validation_checks=validation_checks,
            warnings=warnings,
            errors=errors
        )
    
    def _check_euler_range(self, roll: float, pitch: float, yaw: float) -> bool:
        """Check if Euler angles are within valid range"""
        valid_range = np.pi
        try:
            return all(-valid_range <= float(angle) <= valid_range for angle in [roll, pitch, yaw])
        except (TypeError, ValueError):
            return False
    
    def _check_physical_plausibility(self, roll: float, pitch: float, yaw: float, class_name: str) -> bool:
        """Check if pose angles make physical sense for the object type"""
        try:
            constraints = self.object_constraints.get(class_name, self.object_constraints['default'])
            
            max_roll = constraints['max_roll']
            max_elev = constraints['max_elev']
            
            # Check roll (rotation around Z-axis)
            if abs(float(roll)) > max_roll:
                return False
            
            # Check pitch/elevation (rotation around X-axis)
            if abs(float(pitch)) > max_elev:
                return False
            
            # Yaw can be any angle (rotation around Y-axis)
            return True
        except (TypeError, ValueError):
            return False
    
    def _check_extreme_angles(self, roll: float, pitch: float, yaw: float) -> bool:
        """Check if any angles are extremely large (potential errors)"""
        extreme_threshold = np.pi * 0.9  # 90% of π
        try:
            return all(abs(float(angle)) <= extreme_threshold for angle in [roll, pitch, yaw])
        except (TypeError, ValueError):
            return False
    
    def _check_non_identity_pose(self, roll: float, pitch: float, yaw: float) -> bool:
        """Check if pose is not identity (all angles near zero)"""
        identity_threshold = 0.01  # Very small threshold
        try:
            return any(abs(float(angle)) > identity_threshold for angle in [roll, pitch, yaw])
        except (TypeError, ValueError):
            return False
    
    def _get_min_confidence(self, class_name: str) -> float:
        """Get minimum confidence threshold for object class"""
        constraints = self.object_constraints.get(class_name, self.object_constraints['default'])
        return constraints['min_confidence']
    
    def validate_scene_poses(self, detections: List[Dict[str, Any]]) -> Dict[str, PoseValidationResult]:
        """Validate poses for all objects in a scene"""
        results = {}
        
        for det in detections:
            if 'pcd_orient_bbox' in det and 'eulers' in det['pcd_orient_bbox']:
                # Try to get confidence from different possible sources
                confidence = None
                if 'confidence' in det:
                    confidence = det['confidence']
                elif 'pcd_orient_bbox' in det and 'confidence' in det['pcd_orient_bbox']:
                    confidence = det['pcd_orient_bbox']['confidence']
                
                result = self.validate_pose(det, confidence)
                results[det.get('class_name', 'unknown')] = result
        
        return results
    
    def generate_validation_summary(self, validation_results: Dict[str, PoseValidationResult]) -> Dict[str, Any]:
        """Generate a summary of pose validation results"""
        total_objects = len(validation_results)
        valid_poses = sum(1 for result in validation_results.values() if result.is_valid)
        total_warnings = sum(len(result.warnings) for result in validation_results.values())
        total_errors = sum(len(result.errors) for result in validation_results.values())
        
        # Per-class statistics
        class_stats = {}
        for class_name, result in validation_results.items():
            if class_name not in class_stats:
                class_stats[class_name] = {'total': 0, 'valid': 0, 'warnings': 0, 'errors': 0}
            
            class_stats[class_name]['total'] += 1
            if bool(result.is_valid):
                class_stats[class_name]['valid'] += 1
            class_stats[class_name]['warnings'] += len(result.warnings)
            class_stats[class_name]['errors'] += len(result.errors)
        
        return {
            'total_objects': total_objects,
            'valid_poses': valid_poses,
            'invalid_poses': total_objects - valid_poses,
            'validation_rate': valid_poses / total_objects if total_objects > 0 else 0.0,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'class_statistics': class_stats
        }
