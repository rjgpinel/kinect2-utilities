import numpy as np
import cv2


def unproject(v, fx, fy, cx, cy):
    """
    Transforms 2d camera pixel coordinates and depth to 3d wrt camera frame
    """
    v = v.copy()
    d = v[..., 2]
    v[..., 0] = (v[..., 0] - cx) * d / fx
    v[..., 1] = (v[..., 1] - cy) * d / fy
    return v


def project(v, fx, fy, cx, cy):
    """
    Transforms 3d coordinates wrt camera frame to 2d camera pixel coordinates
    """
    v = v.copy()
    d = v[..., 2]
    v[..., 0] = v[..., 0] * fx / d + cx
    v[..., 1] = v[..., 1] * fy / d + cy
    # v[..., :2] = v[..., :2].round()
    return v


def depth_to_points(depth):
    h, w = depth.shape
    pts = []
    for i in range(h):
        for j in range(w):
            pts.append((j, i, depth[i, j]))
    pts = np.array(pts)
    depth_info = pts[..., 2] > 0
    pts = pts[depth_info]
    return pts, depth_info
