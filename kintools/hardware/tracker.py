import openvr
import numpy as np


devices_sn = {
    "lighthouse": "LHB-F05C4950",
    "camera_tracker": "LHR-01B5FE0B",
    "free_tracker": "LHR-0DBB917B",
}


class VR:
    def __init__(self):
        self.vr = openvr.init(openvr.VRApplication_Background)
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
        Returns a list of tracked device, only return trackers and tracking reference here
        """
        vr = self.vr
        poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseRawAndUncalibrated,
            0,
            openvr.k_unMaxTrackedDeviceCount,
        )
        tracked_devices = {}
        print("Connected devices:")
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bDeviceIsConnected:
                device_class = vr.getTrackedDeviceClass(i)
                device_name = self.devices_name[device_class]
                serial_number = self.vr.getStringTrackedDeviceProperty(
                    i, openvr.Prop_SerialNumber_String
                )
                str_track = "track"
                tracker = TrackedDevice(vr, i, device_name)
                tracked_devices[serial_number] = tracker
                print("{} {} {}".format(i, device_name, serial_number))
        return tracked_devices


class TrackedDevice:
    def __init__(self, vr, index, device_name):
        self.vr = vr
        self.index = index
        self.device_name = device_name

    def get_serial_number(self):
        return self.vr.getStringTrackedDeviceProperty(
            self.index, openvr.Prop_SerialNumber_String
        )

    def get_battery_percent(self):
        return self.vr.getFloatTrackedDeviceProperty(
            self.index, openvr.Prop_DeviceBatteryPercentage_Float
        )

    def get_pose(self):
        """
        pose gives the transformation tracked_device -> world
        invert the pose to get world -> tracked_device
        """
        poses = self.vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseRawAndUncalibrated,
            0,
            openvr.k_unMaxTrackedDeviceCount,
        )
        pose = poses[self.index]
        if pose.bPoseIsValid:
            abs_pose = pose.mDeviceToAbsoluteTracking
            abs_pose = self._pose_to_se3(abs_pose)
            return np.linalg.inv(abs_pose)
        else:
            return None

    def _pose_to_se3(self, pose):
        T = np.zeros((4, 4))
        for i in range(3):
            T[i] = pose[i]
        T[3, 3] = 1
        return T
