import numpy as np
import yaml
import os


def depth_to_camera(depth, fx, fy, cx, cy):
    h, w = depth.shape
    pts = []
    for i in range(h):
        for j in range(w):
            Z = depth[i, j]
            if Z > 0:
                pt = np.array([-(j - cx) * Z / fx, -(i - cy) * Z / fy, Z, 1])
                pts.append(pt)
    pts = np.array(pts)
    return pts


def depth_to_world(depth, fx, fy, cx, cy, extrinsic):
    pts = depth_to_camera(depth, fx, fy, cx, cy)
    pts = pts.dot(extrinsic.T)[:, :3]

    return pts

def get_intrinsics_kin2(depth):
    base_dir = os.path.dirname(__file__)
    params = yaml.load(
        open(os.path.join(base_dir, "params.yml"), "r"),
        Loader=yaml.FullLoader,
    )
    params = params["depth"]
    fovx, fovy = np.pi / 180 * np.array(params["fov"])
    h, w = depth.shape
    fx = w / (2 * np.tan(fovx / 2))
    fy = h / (2 * np.tan(fovy / 2))
    cx, cy = params["optical_center"]
    return fx, fy, cx, cy
