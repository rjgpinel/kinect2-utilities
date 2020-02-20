import click
import numpy as np
import open3d as o3d


@click.command()
@click.argument("filename")
def main(filename):
    pts = np.loadtxt(filename)
    # C = [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255)
    # pcd.transform(C)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
