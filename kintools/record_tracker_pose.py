import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from kintools.tracker import VR

vr = VR()
tracker = vr.track_devices()[0]
print("Recording....")
poses = []
for i in tqdm(range(50)):
    pose = tracker.get_pose()
    poses.append(pose)
    time.sleep(0.1)
# poses = np.array(poses)
ref_pose_inv = np.linalg.inv(poses[0])
poses = [ref_pose_inv.dot(p) for p in poses]
points = np.array([pose[:-1, -1] for pose in poses])


def plot(ax, X, Y, Z):
    scat = ax.scatter(X, Y, Z, c=np.arange(X.shape[0]) / X.shape[0])

    max_range = (
        np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    )

    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# ax.set_aspect("equal")
plot(ax, points[:, 0], points[:, 1], points[:, 2])
plt.show()
# print(trackers[0].get_battery_percent())
# poses = []  # will be populated with proper type after first call
# for i in range(100):
#     import pudb; pudb.set_trace()
#     poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
#     hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
#     print(hmd_pose.mDeviceToAbsoluteTracking)
#     sys.stdout.flush()
#     time.sleep(0.2)
