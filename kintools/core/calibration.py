import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def generate_board(write=False, show=False):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard_create(5, 7, 0.06, 0.04, aruco_dict)
    imboard = board.draw((500, 700), 50, 1)
    if write:
        cv2.imwrite("chessboard.tiff", imboard)
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(imboard, cmap=mpl.cm.gray, interpolation="nearest")
        ax.axis("off")
        plt.show()
    return aruco_dict, board


def extract_corners_chessboard(im, inner_corners_shape):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    c0, c1 = inner_corners_shape
    ret, corners = cv2.findChessboardCorners(gray, (c0, c1), None)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    corners_subpix = []
    if ret:
        # subpixel corners detection
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return corners_subpix, ret


def get_pose_chessboard(corners, inner_corners_shape, mtx, dist):
    c0, c1 = inner_corners_shape
    objp = np.zeros((c0 * c1, 3), np.float32)
    objp[:, :2] = np.mgrid[0:c0, 0:c1].T.reshape(-1, 2)
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
    return rvecs, tvecs, ret


def draw_referential(im, corners, rvecs, tvecs, mtx, dist):
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)
    impts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

    corner = tuple(corners[0].ravel())
    im = cv2.line(im, corner, tuple(impts[0].ravel()), (255, 0, 0), 2)
    im = cv2.line(im, corner, tuple(impts[1].ravel()), (0, 255, 0), 2)
    im = cv2.line(im, corner, tuple(impts[2].ravel()), (0, 0, 255), 2)

    return im


def extract_corners_charuco(im, aruco_dict, board):
    """
    Charuco base pose estimation.
    """
    all_corners = []
    all_ids = []
    # SUB PIXEL CORNER DETECTION CRITERION
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = im
    corners, ids, rejected_im_points = cv2.aruco.detectMarkers(gray, aruco_dict)

    if len(corners) > 0:
        #     # SUB PIXEL DETECTION
        #     for corner in corners:
        #         cv2.cornerSubPix(
        #             gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria
        #         )
        ret, corners, ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
    if corners is None:
        corners = []
        ids = []

    imsize = gray.shape
    return corners, ids, imsize
