import numpy as np
import yaml
import os

from pylibfreenect2 import IrCameraParams, ColorCameraParams


def get_intrinsics_kin2():
    base_dir = os.path.dirname(__file__)
    params = yaml.load(
        open(os.path.join(base_dir, "kinect2.yml"), "r"), Loader=yaml.FullLoader,
    )
    intrinsics = {}
    for key in ["rgb", "depth"]:
        d = params[key]
        fx, fy = d["focal_length"]
        cx, cy = d["optical_center"]
        intrinsics[key] = (fx, fy, cx, cy, d["dist_coeffs"])
    return intrinsics


def get_calibration():
    base_dir = os.path.dirname(__file__)
    folder = os.path.join(base_dir, "../calibration/grenoble_kinect2")

    ir = yaml.load(
        open(os.path.join(folder, "calib_ir.yaml"), "r"), Loader=yaml.FullLoader,
    )
    color = yaml.load(
        open(os.path.join(folder, "calib_color.yaml"), "r"), Loader=yaml.FullLoader,
    )
    depth = yaml.load(
        open(os.path.join(folder, "calib_depth.yaml"), "r"), Loader=yaml.FullLoader,
    )
    pose = yaml.load(
        open(os.path.join(folder, "calib_pose.yaml"), "r"), Loader=yaml.FullLoader,
    )
    for params in [color, ir, depth, pose]:
        for k, v in params.items():
            params[k] = np.array(v)
        if "cameraMatrix" in params:
            params["cameraMatrix"] = params["cameraMatrix"].reshape(3, 3)
        if "rotation" in params:
            params["rotation"] = params["rotation"].reshape(3, 3)
    return ir, color, depth, pose


def get_default_registration_params():
    ir_dict = {
        "cx": 257.5567932128906,
        "cy": 207.3726043701172,
        "fx": 364.73779296875,
        "fy": 364.73779296875,
        "k1": 0.0904708281159401,
        "k2": -0.26997110247612,
        "k3": 0.09681785851716995,
        "p1": 0.0,
        "p2": 0.0,
    }
    color_dict = {
        "cx": 959.5,
        "cy": 539.5,
        "fx": 1081.3720703125,
        "fy": 1081.3720703125,
        "mx_x0y0": 0.17782199382781982,
        "mx_x0y1": -0.00018228229600936174,
        "mx_x0y2": -0.0001405157963745296,
        "mx_x0y3": 4.447028914000839e-06,
        "mx_x1y0": 0.6551164984703064,
        "mx_x1y1": 0.0004178670060355216,
        "mx_x1y2": 0.00019438070012256503,
        "mx_x2y0": 0.00016134009638335556,
        "mx_x2y1": 1.658730980125256e-05,
        "mx_x3y0": 0.00019193769549019635,
        "my_x0y0": 0.0026888330467045307,
        "my_x0y1": 0.6527568101882935,
        "my_x0y2": 0.0004983125254511833,
        "my_x0y3": 0.000874068820849061,
        "my_x1y0": 0.0005328227998688817,
        "my_x1y1": 7.990533777046949e-05,
        "my_x1y2": 6.59358011034783e-06,
        "my_x2y0": -2.7720190701074898e-05,
        "my_x2y1": 0.00030676659662276506,
        "my_x3y0": -1.2843030162912328e-05,
        "shift_d": 863.0,
        "shift_m": 52.0,
    }
    ir_params = IrCameraParams(**ir_dict)
    color_params = ColorCameraParams(**color_dict)
    return ir_params, color_params


def set_camera_params_from_calibration(
    ir_params, color_params, ir_calib, color_calib, depth_calib, pose_calib
):
    ir_cam = ir_calib["cameraMatrix"]
    ir_dist = ir_calib["distortionCoefficients"]
    color_cam = color_calib["cameraMatrix"]
    color_dist = color_calib["distortionCoefficients"]

    ir_params.cx = ir_cam[0, 2]
    ir_params.cy = ir_cam[1, 2]
    ir_params.fx = ir_cam[0, 0]
    ir_params.fy = ir_cam[1, 1]
    ir_params.k1 = ir_dist[0]
    ir_params.k2 = ir_dist[1]
    ir_params.p1 = ir_dist[2]
    ir_params.p2 = ir_dist[3]
    ir_params.k3 = ir_dist[4]

    color_params.cx = color_cam[0, 2]
    color_params.cy = color_cam[1, 2]
    color_params.fx = color_cam[0, 0]
    color_params.fy = color_cam[1, 1]
    # color_params.shift_d = depth_calib["depthShift"]
    return ir_params, color_params
