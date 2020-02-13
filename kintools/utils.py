import numpy as np
import yaml
import os


def get_intrinsics_kin2(depth):
    base_dir = os.path.dirname(__file__)
    params = yaml.load(
        open(os.path.join(base_dir, "params.yml"), "r"), Loader=yaml.FullLoader,
    )
    params = params["depth"]
    fovx, fovy = np.pi / 180 * np.array(params["fov"])
    h, w = depth.shape
    fx = w / (2 * np.tan(fovx / 2))
    fy = h / (2 * np.tan(fovy / 2))
    cx, cy = params["optical_center"]
    return fx, fy, cx, cy
