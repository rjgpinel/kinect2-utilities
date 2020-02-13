import click
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import kintools.utils as utils


def preprocess(x):
    x /= 1000
    x[x > 2.7] = 0
    return x


def depth_to_world(depth, fx, fy, cx, cy, extrinsic):
    pts = depth_to_camera(depth, fx, fy, cx, cy)
    pts = pts.dot(extrinsic.T)[:, :3]

    return pts


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


@click.command()
@click.option("--input", "-i", default="")
@click.option("--output", "-o", default="")
def main(input, output):
    depth = np.load(input)
    depth = preprocess(depth)
    fx, fy, cx, cy = utils.get_intrinsics_kin2(depth)
    pts = depth_to_camera(depth, fx, fy, cx, cy)[:, :3]
    np.savetxt(output, pts)
    plt.figure()
    plt.imshow(depth)
    plt.show()


if __name__ == "__main__":
    main()
