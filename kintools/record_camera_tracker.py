import numpy as np
import os
import pickle as pkl
import cv2

from kintools.hardware.tracker import VR, devices_sn
from kintools.hardware.kinect2 import Kinect2
from kintools.core import calibration


# camera
camera = Kinect2("/home/rstrudel/code/kinect2-utilities/calibration/paris_kinect2")
ir_params = camera.calibration["ir"]
camera_matrix = ir_params["cameraMatrix"]
dist = ir_params["distortionCoefficients"]
pattern_shape = (7, 5)

# tracker
vr = VR()
tracked_devices = vr.track_devices()
lighthouse = tracked_devices[devices_sn["lighthouse"]]
tracker = tracked_devices[devices_sn["camera_tracker"]]

savedir_placement = "snap_cam_tracker"
savedir_depth = "snap_depth_tracker"


def print_T(T_cam, T_tracker):
    print("-> camera")
    print(T_cam)
    print(np.linalg.norm(T_cam[:3, 3]))
    print("-> tracker")
    print(T_tracker)
    print(np.linalg.norm(T_tracker[:3, 3]))


print("Recording....")
np.set_printoptions(suppress=True)
idx_placement = 36
idx_depth = 0
T_cams = []
T_trackers = []
while True:
    T_lighthouse = lighthouse.get_pose()
    T_tracker = tracker.get_pose()
    T_lighthouse_tracker = T_tracker @ np.linalg.inv(T_lighthouse)

    color, depth, ir = camera.snapshot()

    corners, ret = calibration.extract_corners_chessboard(ir, pattern_shape)
    rvecs, tvecs = None, None
    T_world_cam = None
    if ret:
        rvecs, tvecs, ret = calibration.get_pose_chessboard(
            corners, pattern_shape, camera_matrix, dist
        )
    if ret:
        R, _ = cv2.Rodrigues(rvecs[:, 0])
        T_world_cam = np.zeros((4, 4))
        T_world_cam[:3, :3] = R
        T_world_cam[:3, 3] = tvecs[:, 0] * 3.5 / 100
        T_world_cam[3, 3] = 1
        calibration.draw_referential(ir, corners, rvecs, tvecs, camera_matrix, dist)

    # cv2.imshow("ir", ir)
    depth[depth > 2000] = 0
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_norm = (255 * depth_norm).astype(np.uint8)
    depth_cmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    cv2.imshow("depth", depth_cmap)
    # cv2.imshow("color", color)
    # cv2.imshow("depth", depth_norm)

    T_tracker = T_lighthouse_tracker
    # T_tracker = T_tracker
    T_cam = T_world_cam
    # keyboard handling
    pressed_key = cv2.waitKey(1) & 0xFF
    if pressed_key == 32 and len(corners) > 0:
        # space
        if len(T_cams) > 0:
            print("relative")
            print_T(
                T_cam @ np.linalg.inv(T_cams[-1]),
                T_tracker @ np.linalg.inv(T_trackers[-1]),
            )
        else:
            print_T(T_cam, T_tracker)
        print("")
        snap = {"camera": T_cam, "tracker": T_tracker}
        snapshot_path = os.path.join(
            savedir_placement, "snap{}.pkl".format(idx_placement)
        )
        pkl.dump(snap, open(snapshot_path, "wb"))
        print("Record tracker/camera pose: {}".format(idx_placement))
        idx_placement += 1
        T_cams.append(T_cam)
        T_trackers.append(T_tracker)
    elif pressed_key == ord("p"):
        if T_cam is None:
            T_cam = np.eye(4)
        # if len(T_cams) > 0:
        #     print("relative")
        #     print_T(T_cam @ np.linalg.inv(T_cams[-1]), T_tracker @ np.linalg.inv(T_trackers[-1]))
        # else:
        print_T(T_cam, T_tracker)
        print("")
        T_cams.append(T_cam)
        T_trackers.append(T_tracker)
    elif pressed_key == ord("s"):
        # snapshot
        snap = {"tracker": T_tracker, "depth": depth}
        snapshot_path = os.path.join(savedir_depth, "snap{}.pkl".format(idx_depth))
        pkl.dump(snap, open(snapshot_path, "wb"))
        print("Record depth and tracker pose: {}".format(idx_depth))
        idx_depth += 1
    elif pressed_key == ord("q"):
        break
