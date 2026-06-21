from datetime import datetime
from itertools import groupby
import logging
import os
from typing import Any, Dict

import numpy as np

AnnotType = Dict[str, Any]

COLORS_8 = [
    [119, 170, 221],
    [153, 221, 255],
    [68, 187, 153],
    [187, 204, 51],
    [170, 170, 0],
    [238, 221, 136],
    [238, 136, 102],
    [255, 170, 187]]
COLORS_8_F = np.array(COLORS_8) / 255.0


class SkipSampleException(Exception):
    """Raise this exception if current sample should be skipped for various
    reasons.
    """
    pass


def setup_logging(save_path=None):
    if save_path is None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        os.path.makedirs(save_path, exist_ok=True)
        dt = datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(save_path, f'log_{dt}.txt'),
            filemode='w',
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)
        return logging.getLogger("").handlers[0].baseFilename


def serialize_pcd(pcd, save_pcd=False):
    pcd_points = np.asarray(pcd.points)
    size1 = np.max(pcd_points[:, 0]) - np.min(pcd_points[:, 0])
    size2 = np.max(pcd_points[:, 1]) - np.min(pcd_points[:, 1])
    size3 = np.max(pcd_points[:, 2]) - np.min(pcd_points[:, 2])
    if save_pcd:
        return dict(
            type='open3d.cuda.pybind.geometry.PointCloud',
            points=np.asarray(pcd.points).tolist(),
            colors=np.asarray(pcd.colors).tolist(),
            size=[size1, size2, size3])
    else:
        return dict(
            type='open3d.cuda.pybind.geometry.PointCloud',
            size=[size1, size2, size3])


def mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def serialize(annot: AnnotType, save_pcd: bool = False) -> AnnotType:
    if 'vis' in annot:
        annot.pop('vis')
    
    # Handle pose validation summary if present
    if 'pose_validation_summary' in annot:
        # This field contains validation statistics and should be preserved
        pass
    
    # Check if scene_3d_info exists before processing it
    if 'scene_3d_info' in annot and isinstance(annot['scene_3d_info'], dict):
        for k in list(annot['scene_3d_info'].keys()):
            try:
                if k in ['intrinsic', 'pf_R', 'min_y']:
                    if hasattr(annot['scene_3d_info'][k], 'tolist'):
                        annot['scene_3d_info'][k] = annot['scene_3d_info'][k].tolist()
                elif k in ['pcd', 'pcd_cano']:
                    # annot['scene_3d_info'][k] = dict(
                    #     type='open3d.cuda.pybind.geometry.PointCloud',
                    #     points=np.asarray(annot['scene_3d_info'][k].points).tolist(),
                    #     colors=np.asarray(annot['scene_3d_info'][k].colors).tolist() if annot['scene_3d_info'][k].has_colors() else None,
                    #     normals=np.asarray(annot['scene_3d_info'][k].normals).tolist() if annot['scene_3d_info'][k].has_normals() else None)
                    annot['scene_3d_info'][k] = serialize_pcd(annot['scene_3d_info'][k], save_pcd=save_pcd)
                else:
                    # Skip unknown keys instead of raising error
                    continue
            except Exception as e:
                # Log the error and continue with other keys
                print(f"Warning: Failed to serialize scene_3d_info key '{k}': {e}")
                continue
    
    # Check if detections exist before processing them
    if 'detections' in annot and isinstance(annot['detections'], list):
        for obj in annot['detections']:
            if not isinstance(obj, dict):
                continue
            for k in list(obj.keys()):
                try:
                    if k in ['class_name', 'confidence', 'box_area', 'area', 'class_id', 'pose', 'object_name', 'numbered_label', 'pose_validation']:
                        pass
                    elif k in ['xyxy', 'pcd_center', 'pcd_cano_center', 'left', 'front', 'up']:
                        # Check if data is already a list before calling .tolist()
                        if hasattr(obj[k], 'tolist'):
                            obj[k] = obj[k].tolist()
                        # If it's already a list, leave it as is
                    elif k in ['mask', 'mask_subtracted']:
                        obj[k+'_rle'] = mask_to_rle(obj[k])
                        obj.pop(k)
                    elif k in ['pcd', 'pcd_cano']:
                        # obj[k] = dict(
                        #     type='open3d.cuda.pybind.geometry.PointCloud',
                        #     points=np.asarray(obj[k].points).tolist(),
                        #     colors=np.asarray(obj[k].colors).tolist() if obj[k].has_colors() else None,
                        #     normals=np.asarray(obj[k].normals).tolist() if obj[k].has_normals() else None)
                        obj[k] = serialize_pcd(obj[k], save_pcd=save_pcd)
                    elif k in ['pcd_axis_bbox', 'pcd_cano_axis_bbox', 'pcd_orient_bbox', 'pcd_cano_orient_bbox']:
                        if isinstance(obj[k], dict):
                            obj[k] = {_k: obj[k][_k].tolist() if hasattr(obj[k][_k], 'tolist') else obj[k][_k] for _k in obj[k]}
                    else:
                        # Skip unknown keys instead of raising error
                        continue
                except Exception as e:
                    # Log the error and continue with other keys
                    print(f"Warning: Failed to serialize detection key '{k}': {e}")
                    continue
    
    return annot