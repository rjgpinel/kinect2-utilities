import openvr
import numpy as np


class VR:
    def __init__(self):
        self.vr = openvr.init(openvr.VRApplication_Scene)
        self.devices_name = {
            openvr.TrackedDeviceClass_Controller: "controller",
            openvr.TrackedDeviceClass_GenericTracker: "tracker",
            openvr.TrackedDeviceClass_HMD: "head",
            openvr.TrackedDeviceClass_TrackingReference: "tracking_reference",
        }

    def __del__(self):
        openvr.shutdown()

    def track_devices(self):
        """
        Returns a list of tracked device, only return trackers here
        """
        vr = self.vr
        poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        trackers = []
        print("Connected devices:")
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bDeviceIsConnected:
                device_class = vr.getTrackedDeviceClass(i)
                device_name = self.devices_name[device_class]
                if device_name == "tracker":
                    tracker = TrackedDevice(vr, i)
                    trackers.append(tracker)
                    print("{} {} : track".format(i, device_name))
                else:
                    print("{} {} : discard".format(i, device_name))
        return trackers


class TrackedDevice:
    def __init__(self, vr, index):
        self.vr = vr
        self.index = index

    def get_battery_percent(self):
        return self.vr.getFloatTrackedDeviceProperty(
            self.index, openvr.Prop_DeviceBatteryPercentage_Float
        )

    def get_pose(self):
        poses = self.vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        pose = poses[self.index]
        if pose.bPoseIsValid:
            abs_pose = pose.mDeviceToAbsoluteTracking
            abs_pose = self._pose_to_se3(abs_pose)
            return abs_pose
        else:
            return None

    def _pose_to_se3(self, pose):
        T = np.zeros((4, 4))
        for i in range(3):
            T[i] = pose[i]
        T[3, 3] = 1
        return T
