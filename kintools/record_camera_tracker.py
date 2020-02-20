import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt

from kintools.tracker import VR
from kintools.kinect2 import Kinect2


camera = Kinect2()
vr = VR()
tracker = vr.track_devices()[0]
savedir = "snapshots"

print("Recording....")
idx = 0
while True:
    pose = tracker.get_pose()
    color, depth = camera.snapshot()
    geo_snapshot = {"color": color, "depth": depth, "pose": pose}
    snapshot_path = os.path.join(savedir, "geosnapshot{}.pkl".format(idx))
    pkl.dump(geo_snapshot, open(snapshot_path, "wb"))

    plt.subplot(2, 2, 1)
    plt.imshow(color)
    plt.title("color {}".format(idx))

    plt.subplot(2, 2, 2)
    plt.imshow(depth)
    plt.title("depth {}".format(idx))

    plt.show()
    idx += 1
