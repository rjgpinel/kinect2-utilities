import click
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from pylibfreenect2 import Registration, Frame, FrameType

import kintools.utils as utils


def preprocess(color, d):
    color = color[..., :3]
    color = color[..., ::-1]
    # d /= 1000
    # d[d > 2.0] = 0
    # d[d < 0.2] = 0
    return color, d


def homogeneous(x):
    o = np.ones((x.shape[0], 1))
    return np.concatenate((x, o), axis=1)


def unproject(v, intrinsics):
    """
    Transforms 2d camera pixel coordinates and depth to 3d wrt camera frame
    """
    fx, fy, cx, cy, _ = intrinsics
    v = v.copy()
    d = v[..., 2]
    v[..., 0] = (v[..., 0] - cx) * d / fx
    v[..., 1] = (v[..., 1] - cy) * d / fy
    return v


def project(v, intrinsics):
    """
    Transforms 3d coordinates wrt camera frame to 2d camera pixel coordinates
    """
    fx, fy, cx, cy, _ = intrinsics
    v = v.copy()
    d = v[..., 2]
    v[..., 0] = v[..., 0] * fx / d + cx
    v[..., 1] = v[..., 1] * fy / d + cy
    # v[..., :2] = v[..., :2].round()
    return v


def color_to_ir():
    T = np.array(
        [
            [0.999975, 0.00498997, 0.00500991, 0.053],
            [-0.00499992, 0.99998555, 0.00197497, 0.001],
            [-0.00499998, -0.00199997, 0.9999855, 0.004],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return T


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


def color_to_points(color, depth_info):
    h, w, c = color.shape
    pts = []
    for i in range(h):
        for j in range(w):
            pts.append(color[i, j])
    pts = np.array(pts)
    pts = pts[depth_info]
    return pts


# def colorize(color, pts_2d_color):
#     h, w, _ = color.shape
#     pts_colorized = []
#     for point in pts_2d_color:
#         x, y, z = point
#         xr, yr, zr = point.round().astype(np.int)
#         if yr > 0 and yr < h and xr > 0 and xr < w:
#             r, g, b = color[yr, xr]
#             pts_colorized.append((x, y, z, r, g, b))
#     pts_colorized = np.array(pts_colorized)
#     return pts_colorized


def colorize(pts_3d_depth, aligned_2d_color):
    pts_colorized = []
    for point, color in zip(pts_3d_depth, aligned_2d_color):
        x, y, z = point
        r, g, b = color
        if np.linalg.norm(np.array(color)) > 1e-2:
            pts_colorized.append((x, y, z, r, g, b))
    pts_colorized = np.array(pts_colorized)
    return pts_colorized


def register(registration, color, depth):
    color = color.copy(order="C")
    depth = depth.copy(order="C")
    color_fr = Frame(512, 424, 3, numpy_array=color)
    depth_fr = Frame(512, 424, 1)
    color_fr.color = Frame(
        numpy_array=color, frame_type=FrameType.Color, bytes_per_pixel=3
    )
    depth_fr = Frame(numpy_array=depth, frame_type=FrameType.Depth)
    undistorted_fr = Frame(512, 424, 4)
    registered_fr = Frame(512, 424, 4)
    registration.apply(color_fr, depth_fr, undistorted_fr, registered_fr)
    registered = registered_fr.asarray(dtype=np.uint8).copy()
    return registered


@click.command()
@click.option("--depth", "-d", default="")
@click.option("--color", "-color", default="")
@click.option("--output", "-o", default="")
def main(depth, color, output):
    intrinsics = utils.get_intrinsics_kin2()
    ir_params, color_params = utils.get_default_registration_params()
    calibration = utils.get_calibration()
    utils.set_camera_params_from_calibration(ir_params, color_params, *calibration)
    registration = Registration(ir_params, color_params)

    color = np.load(color)
    depth = np.load(depth)
    color, depth = preprocess(color, depth)
    registered = register(registration, color, depth)
    plt.figure()
    plt.imshow(registered)
    plt.show()
    # T_depth_to_color = np.linalg.inv(color_to_ir())
    pts_2d_depth, depth_info = depth_to_points(depth)
    # pts_2d_color = color_to_points(color, depth_info)

    pts_3d_depth = unproject(pts_2d_depth, intrinsics["depth"])
    # colorized = colorize(pts_3d_depth, pts_2d_color)
    # pts_3d_depth = homogeneous(pts_3d_depth)
    # pts_3d_color = pts_3d_depth.dot(T_depth_to_color.T)[:, :3]
    # pts_2d_color = project(pts_3d_color, intrinsics["color"])
    # colorized = pts_2d_color
    # colorized = colorize(color, pts_2d_color)
    # colorized[:, :3] = unproject(colorized[:, :3], intrinsics["color"])
    # registration = Registration(ir_params, color_params)

    np.savetxt(output, pts_3d_depth)
    # np.savetxt(output, colorized)
    plt.figure()
    plt.imshow(depth)
    plt.show()


if __name__ == "__main__":
    main()
