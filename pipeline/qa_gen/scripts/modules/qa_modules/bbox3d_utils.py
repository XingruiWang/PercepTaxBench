"""
3D Bounding Box and Projection Utilities

Functions for working with 3D bounding boxes and projecting them to 2D image planes.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_bbox3d(center, extent, rotation_matrix=None):
    """Get 3D bounding box corners and edges (could be oriented bounding box).

    Args:
        center (np.ndarray): Center of the bounding box (3,).
        extent (np.ndarray): Extent of the bounding box (3,).
        rotation_matrix (np.ndarray): Rotation matrix (3, 3). Default is None. If not None, the corners will be corners for oriented bounding box.

    Returns:
        corners (np.ndarray): 3D bounding box corners (8, 3).
        edges (np.ndarray): 3D bounding box edges (12, 2).
    """
    corners = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ]
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    corners = corners * extent
    if rotation_matrix is not None:
        corners = (rotation_matrix @ corners.T).T
    corners = corners + center
    return corners, edges


def fx_fy_from_fovx(fovx_deg, W, H):
    """Calculate focal lengths from horizontal FOV"""
    fovx = np.deg2rad(fovx_deg)
    fx = 0.5 * W / np.tan(0.5 * fovx)
    fovy = 2.0 * np.arctan((H / float(W)) * np.tan(0.5 * fovx))
    fy = 0.5 * H / np.tan(0.5 * fovy)
    return fx, fy


def project_3d_to_2d(corners, c2w, fov, W, H):
    """Project 3D points to 2D image plane.

    Args:
        corners (np.ndarray): 3D points (N, 3).
        c2w (np.ndarray): Camera-to-world transformation matrix (4, 4).
        fov (float): Horizontal field of view in degrees.
        W (int): Image width.
        H (int): Image height.

    Returns:
        uv (np.ndarray): 2D projected points (N, 2).
        in_front (np.ndarray): Boolean mask indicating if points are in front of the camera (N,).
    """
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    pc = (w2c @ pts_h.T).T
    Xc, Yc, Zc = pc[:, 0], pc[:, 1], pc[:, 2]

    in_front = Xc > 0

    fx, fy = fx_fy_from_fovx(fov, W, H)
    cx, cy = 0.5 * W, 0.5 * H

    u = fx * (Yc / Xc) + cx
    v = fy * (-Zc / Xc) + cy

    uv = np.stack([u, v], axis=-1).reshape(-1, 2)
    in_front = in_front.reshape(corners.shape[:-1])
    return uv, in_front


def rot_mat(rotation):
    """Convert rotation from (XZY) Euler angles in degrees to rotation matrix.

    Args:
        rotation (list or np.ndarray): Rotation angles in degrees [rx, ry, rz].

    Returns:
        np.ndarray: Rotation matrix (3, 3).
    """
    r = R.from_euler("XZY", rotation, degrees=True)
    return r.as_matrix()


def get_camera_matrix_from_annots(camera_annots_data):
    """Extract camera matrix from camera_annots.json
    
    Args:
        camera_annots_data (dict): Contents of camera_annots.json
        
    Returns:
        c2w (np.ndarray): Camera-to-world matrix (4, 4)
        fov (float): Field of view in degrees
        W (int): Image width
        H (int): Image height
    """
    camera_info = camera_annots_data.get('outputs', {})  # Fixed: was 'output', now 'outputs'
    
    # Extract location and rotation
    location = np.array(camera_info.get('location', [0, 0, 0]))
    rotation = np.array(camera_info.get('rotation', [0, 0, 0]))
    
    # Convert rotation to matrix
    R_mat = rot_mat(rotation)
    
    # Build camera-to-world matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R_mat
    c2w[:3, 3] = location
    
    # Get FOV (default 90 degrees if not specified)
    fov = camera_info.get('fov', 90)
    
    # Get image dimensions (default 1920x1080)
    # JSON uses 'width' and 'height', not 'image_width' and 'image_height'
    W = camera_info.get('width', camera_info.get('image_width', 1920))
    H = camera_info.get('height', camera_info.get('image_height', 1080))
    
    return c2w, fov, W, H


def compute_2d_bbox_from_3d_pose(object_pose, camera_annots_data):
    """Compute 2D bounding box from 3D object pose and camera parameters
    
    Args:
        object_pose (dict): Object pose data with location, rotation, bounds
        camera_annots_data (dict): Camera annotations
        
    Returns:
        bbox_2d (list): [x_min, y_min, x_max, y_max]
        visible (bool): Whether object is visible (in front of camera)
    """
    try:
        # Get object 3D parameters
        center = np.array(object_pose.get('location', [0, 0, 0]))
        bounds = object_pose.get('bounds', {})
        extent = np.array(bounds.get('extent', [1, 1, 1]))
        
        # Check if we have OBB (oriented bounding box) or AABB (axis-aligned)
        obb_data = object_pose.get('obb', {})
        if obb_data and obb_data.get('rotation'):
            # Oriented bounding box - use OBB rotation
            rotation = np.array(obb_data.get('rotation', [0, 0, 0]))
            rotation_matrix = rot_mat(rotation)
        else:
            # Axis-aligned bounding box - no rotation
            rotation_matrix = None
        
        # Get 3D bbox corners
        corners_3d, _ = get_bbox3d(center, extent, rotation_matrix)
        
        # Get camera parameters
        c2w, fov, W, H = get_camera_matrix_from_annots(camera_annots_data)
        
        # Project to 2D
        uv, in_front = project_3d_to_2d(corners_3d, c2w, fov, W, H)
        
        # Check if at least some corners are visible
        visible = np.any(in_front)
        
        if not visible:
            return None, False
        
        # Get min/max bounds (only for visible corners)
        visible_uv = uv[in_front]
        
        x_min = np.min(visible_uv[:, 0])
        x_max = np.max(visible_uv[:, 0])
        y_min = np.min(visible_uv[:, 1])
        y_max = np.max(visible_uv[:, 1])
        
        # Clip to image bounds
        x_min = max(0, min(x_min, W - 1))
        x_max = max(0, min(x_max, W - 1))
        y_min = max(0, min(y_min, H - 1))
        y_max = max(0, min(y_max, H - 1))
        
        bbox_2d = [float(x_min), float(y_min), float(x_max), float(y_max)]
        
        return bbox_2d, visible
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error computing 2D bbox from 3D pose: {e}")
        return None, False


def compute_depth_from_3d(location, camera_annots_data):
    """Compute camera-relative depth from 3D location
    
    Args:
        location (np.ndarray): 3D location (3,)
        camera_annots_data (dict): Camera annotations
        
    Returns:
        depth (float): Depth value (distance from camera along X-axis in camera space)
    """
    try:
        c2w, _, _, _ = get_camera_matrix_from_annots(camera_annots_data)
        w2c = np.linalg.inv(c2w)
        
        # Transform point to camera space
        pt_h = np.array([location[0], location[1], location[2], 1])
        pt_camera = (w2c @ pt_h.T)
        
        # X-axis in camera space is depth (forward direction)
        depth = float(pt_camera[0])
        
        return depth
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error computing depth from 3D: {e}")
        return None

