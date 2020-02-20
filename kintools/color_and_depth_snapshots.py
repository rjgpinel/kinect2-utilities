import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import yaml
import os
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

import time
from kintools.utils import get_calibration

try:
    from pylibfreenect2 import OpenGLPacketPipeline

    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline

        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline

        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)


# Create and set logger
# logger = createConsoleLogger(LoggerLevel.Debug)
# setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
ir, color, depth, pose = get_calibration()

ir_params = device.getIrCameraParams()
color_params = device.getColorCameraParams()
# ir_params, color_params = set_params(ir_params, color_params, ir, color, depth, pose)
registration = Registration(ir_params, color_params)

idx_im = 0

viz = True

undistorted_frame = Frame(512, 424, 4)
registered_frame = Frame(512, 424, 4)
save_dir = "kinect_data"

while True:

    print("Waiting for a new frame...")
    frames = listener.waitForNewFrame()

    color = frames["color"]
    depth = frames["depth"]
    color = color.asarray(np.uint8)
    depth = depth.asarray(np.float32)
    np.save(os.path.join(save_dir, "color{}.npy".format(idx_im)), color)
    np.save(os.path.join(save_dir, "depth{}.npy".format(idx_im)), depth)

    # registration.apply(color, depth, undistorted_frame, registered_frame)
    # color = color.asarray()[..., :3].copy()
    # color = color[..., ::-1]
    # color = color[:, ::-1]
    # depth = depth.asarray().copy()
    # depth = depth[:, ::-1]
    # undistorted = undistorted_frame.asarray(dtype=np.float32).copy()
    # undistorted = undistorted[:, ::-1]
    # registered = registered_frame.asarray(dtype=np.uint8).copy()[..., :3]
    # registered = registered[:, ::-1]

    if viz:
        plt.subplot(2, 2, 1)
        plt.imshow(color)
        plt.title("Color")

        plt.subplot(2, 2, 2)
        plt.imshow(depth)
        plt.title("Depth")

        # plt.subplot(2, 2, 3)
        # plt.imshow(undistorted)
        # plt.title("Undistorted")

        # plt.subplot(2, 2, 4)
        # plt.imshow(registered)
        # plt.title("Registered depth")
        plt.show()

    # np.save(os.path.join(save_dir, "color%i.npy" % id), color)
    # np.save(os.path.join(save_dir, "depth%i.npy" % id), depth)
    # np.save(
    #     os.path.join(save_dir, "registered_color%i.npy" % id), registered,
    # )

    idx_im += 1
    listener.release(frames)
    time.sleep(2)

device.stop()
device.close()

sys.exit(0)
