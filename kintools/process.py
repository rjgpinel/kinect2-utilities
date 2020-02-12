import click
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import kintools.utils as utils


def preprocess(x):
    x /= 1000
    x[x > 1.2] = 0
    return x


@click.command()
@click.option("--input", "-i", default="")
@click.option("--output", "-o", default="")
def main(input, output):
    depth = np.load(input)
    depth = preprocess(depth)
    fx, fy, cx, cy = utils.get_intrinsics_kin2(depth)
    pts = utils.depth_to_camera(depth, fx, fy, cx, cy)[:, :3]
    np.savetxt(output, pts)
    plt.figure()
    plt.imshow(depth)
    plt.show()


if __name__ == "__main__":
    main()
