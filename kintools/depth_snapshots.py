import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

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

# Get IR Camera Params
params = device.getIrCameraParams()

print("IR Camera params")
print(
    {
        "fx": params.fx,
        "fy": params.fy,
        "cx": params.cx,
        "cy": params.cy,
        "k1": params.k1,
        "k2": params.k2,
        "k3": params.k3,
        "p1": params.p1,
        "p2": params.p2,
    }
)


listener = SyncMultiFrameListener(FrameType.Depth)

# Register listeners
device.setIrAndDepthFrameListener(listener)

device.start()

id = 0

while True:

    print("Waiting for a new frame...")
    frames = listener.waitForNewFrame()

    depth = frames["depth"]

    np.save("/home/rgarciap/Desktop/depth%i.npy" % id, np.array(depth.asarray()))

    plt.imshow(depth.asarray() / 4500.0)
    plt.show()
    id += 1
    listener.release(frames)

device.stop()
device.close()

sys.exit(0)
