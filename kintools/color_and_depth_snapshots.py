import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

import time

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
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

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
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

id = 0

viz = False

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

while True:

    print('Waiting for a new frame...')
    frames = listener.waitForNewFrame()

    color = frames["color"]
    depth = frames["depth"]

    registration.apply(color, depth, undistorted, registered)

    if viz:

        plt.subplot(2, 2, 1)
        plt.imshow(color.asarray())
        plt.title('Color')

        plt.subplot(2, 2, 2)
        plt.imshow(depth.asarray())
        plt.title('Depth')

        plt.subplot(2, 2, 3)
        plt.imshow(undistorted.asarray(dtype=np.float32))
        plt.title('Undistorted')

        plt.subplot(2, 2, 4)
        plt.imshow(registered.asarray(dtype=np.float32))
        plt.title('Registered depth')
        plt.show()

    np.save('./data/color%i.npy' % id, np.array(color.asarray()))
    np.save('./data/depth%i.npy' % id, np.array(depth.asarray()))
    np.save('./data/registered_depth%i.npy' % id, np.array(registered.asarray(dtype=np.float32)))

    id += 1
    listener.release(frames)
    time.sleep(2)

device.stop()
device.close()

sys.exit(0)
