import click
import pickle as pkl

import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt


def pose_estimation(im, mtx, dist):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )
    # SUB PIXEL DETECTION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    for corner in corners:
        cv2.cornerSubPix(
            gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria
        )

    im_markers = aruco.drawDetectedMarkers(im.copy(), corners, ids)
    # side lenght of the marker in meter
    size_of_marker = 0.0285
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
        corners, size_of_marker, mtx, dist
    )
    length_of_axis = 0.1
    imaxis = aruco.drawDetectedMarkers(im.copy(), corners, ids)

    for i in range(len(tvecs)):
        imaxis = aruco.drawAxis(imaxis, mtx, dist, rvecs[i], tvecs[i], length_of_axis)

    plt.figure(figsize=(15, 10))
    plt.imshow(imaxis)
    plt.grid()
    plt.show()


@click.command()
@click.argument("path")
@click.argument("camera_parameters")
def main(path, camera_parameters):
    params = pkl.load(open(camera_parameters, "rb"))
    camera_matrix = params["camera_matrix"]
    dist_coeffs = params["dist_coeffs"]
    im = cv2.imread(path)
    pose_estimation(im, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    main()
