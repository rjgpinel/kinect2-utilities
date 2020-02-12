"""
Based on
https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html
"""

import click
from tqdm import tqdm
import pickle as pkl

import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl


def generate_board(write=False, show=False):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard_create(7, 5, 1, 0.8, aruco_dict)
    imboard = board.draw((2000, 2000))
    if write:
        cv2.imwrite("chessboard.tiff", imboard)
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(imboard, cmap=mpl.cm.gray, interpolation="nearest")
        ax.axis("off")
        plt.show()
    return aruco_dict, board


def kinect_np_to_jpeg(path, flip=True):
    filenames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npy")]
    for i, filename in enumerate(images):
        im = np.load(filename)[..., :3]
        if flip:
            im = im[:, ::-1]
        cv2.imwrite(os.path.join(path, "color{}.jpg".format(i)), im)


def extract_corners_chessboards(images, aruco_dict, board):
    """
    Charuco base pose estimation.
    """
    print("Processing images for corners detection...")
    all_corners = []
    all_ids = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in tqdm(images):
        # print("processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(
                    gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria
                )
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if (
                res2[1] is not None
                and res2[2] is not None
                and len(res2[1]) > 3
                and decimator % 1 == 0
            ):
                all_corners.append(res2[1])
                all_ids.append(res2[2])

        decimator += 1

    imsize = gray.shape
    return all_corners, all_ids, imsize


def calibrate_camera(all_corners, all_ids, imsize, board):
    """
    Calibrates the camera using the dected corners.
    """
    print("Estimating camera parameters using the detected corners...")

    cameraMatrixInit = np.array(
        [
            [1000.0, 0.0, imsize[0] / 2.0],
            [0.0, 1000.0, imsize[1] / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    distCoeffsInit = np.zeros((5, 1))
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        + cv2.CALIB_RATIONAL_MODEL
        + cv2.CALIB_FIX_ASPECT_RATIO
    )
    # flags = (cv2.CALIB_RATIONAL_MODEL)
    (
        ret,
        camera_matrix,
        distortion_coefficients0,
        rotation_vectors,
        translation_vectors,
        stdDeviationsIntrinsics,
        stdDeviationsExtrinsics,
        perViewErrors,
    ) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9),
    )

    return (
        ret,
        camera_matrix,
        distortion_coefficients0,
        rotation_vectors,
        translation_vectors,
    )


def show_correction(im, mtx, dist):
    plt.figure(figsize=(18, 5))
    frame = cv2.imread(im)
    img_undist = cv2.undistort(frame, mtx, dist, None)
    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title("Raw image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_undist)
    plt.title("Corrected image")
    plt.axis("off")
    plt.show()


@click.command()
@click.argument("path_images")
def main(path_images):
    images = np.array(
        [
            os.path.join(path_images, f)
            for f in os.listdir(path_images)
            if f.endswith(".jpeg")
        ]
    )
    aruco_dict, board = generate_board()
    all_corners, all_ids, imsize = extract_corners_chessboards(
        images, aruco_dict, board
    )
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
        all_corners, all_ids, imsize, board
    )
    params = {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs}
    pkl.dump(params, open("cam_params.pkl", "wb"))
    show_correction(images[0], camera_matrix, dist_coeffs)


if __name__ == "__main__":
    main()
