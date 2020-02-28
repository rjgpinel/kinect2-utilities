import time
import numpy as np

from kintools.hardware.tracker import VR, devices_sn

vr = VR()
tracked_devices = vr.track_devices()

np.set_printoptions(suppress=True)
while True:
    T_lighthouse = tracked_devices[devices_sn["lighthouse"]].get_pose()
    T_tracker = tracked_devices[devices_sn["free_tracker"]].get_pose()
    T_rel = T_tracker @ np.linalg.inv(T_lighthouse)
    print("lighthouse")
    print(T_lighthouse)
    print("tracker")
    print(T_tracker)
    print("rel")
    print(T_rel)
    print(np.linalg.norm(T_rel[:3, 3]))
    # print("lighthouse->tracker")
    # print(T_tracker)
    time.sleep(3)
