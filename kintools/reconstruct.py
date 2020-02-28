import click
import os
import numpy as np
import pickle as pkl
import glob
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

from kintools.core import camera
from kintools.hardware import utils


def preprocess(d):
    d = d / 1000
    d = cv2.medianBlur(d, 5)
    d[d > 1.0] = 0
    # d[d < 0.2] = 0
    return d


def to_homogeneous(x):
    a = np.zeros((x.shape[0], 4))
    a[:, :3] = x
    a[:, 3] = 1
    return a


@click.command()
@click.option("--input_file", "-i", default="")
@click.option("--output_folder", "-o", default="")
def main(input_file, output_folder):
    calibration = utils.load_calibration(
        "/home/rstrudel/code/kinect2-utilities/calibration/paris_kinect2"
    )
    ir_params = calibration["ir"]
    camera_matrix = ir_params["cameraMatrix"]
    dist = ir_params["distortionCoefficients"]
    fx, fy, cx, cy = (
        camera_matrix[0, 0],
        camera_matrix[1, 1],
        camera_matrix[0, 2],
        camera_matrix[1, 2],
    )

    T_cam_tracker = np.array(
        [
            [0.997871, -0.006405, -0.064908, 0.034571],
            [-0.065077, -0.031177, -0.997393, -0.036484],
            [0.004364, 0.999493, -0.031527, 0.019005],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ]
    )
    # T_cam_tracker[:3, :3] = T_cam_tracker[:3, :3]

    files = glob.glob(os.path.join("snap_depth_tracker", "*.pkl"))
    files = sorted(files)
    cloud = np.zeros((0, 3))
    for i, input_file in enumerate(files):
        snap = pkl.load(open(input_file, "rb"))
        depth = snap["depth"]
        T_tracker = snap["tracker"]
        # if i == 0:
        #     T_ref = T_tracker
        T_tracker_base = np.linalg.inv(T_tracker)
        # print(np.linalg.norm(T_tracker_ref[:3, 3]))
        depth = preprocess(depth)

        depth_pts_2d, depth_info = camera.depth_to_points(depth)
        depth_pts_3d = camera.unproject(depth_pts_2d, fx, fy, cx, cy)
        depth_pts_hom = to_homogeneous(depth_pts_3d)
        depth_pts_aligned = np.zeros_like(depth_pts_hom)
        for k, x in enumerate(depth_pts_hom):
            depth_pts_aligned[k] = (T_tracker_base @ T_cam_tracker).dot(x)
            # depth_pts_aligned[k] = x
        cloud = np.concatenate((cloud, depth_pts_aligned[:, :3]))
        np.savetxt(
            os.path.join(output_folder, "res{}.npy".format(i)),
            depth_pts_aligned[:, :3],
        )

    np.savetxt(os.path.join(output_folder, "res_fusion.npy"), cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd_down = pcd.voxel_down_sample(0.02)
    o3d.visualization.draw_geometries([pcd_down])


if __name__ == "__main__":
    main()
