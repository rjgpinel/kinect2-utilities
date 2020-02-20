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

import numpy as np


class Kinect2:
    def __init__(self):
        self.freenect = Freenect2()
        num_devices = self.freenect.enumerateDevices()
        if num_devices == 0:
            raise ValueError("No kinect2 device connected!")

        serial = self.freenect.getDeviceSerialNumber(0)
        self.device = self.freenect.openDevice(serial, pipeline=pipeline)
        self.listener = SyncMultiFrameListener(
            FrameType.Color | FrameType.Ir | FrameType.Depth
        )
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()

    def snapshot(self):
        frames = self.listener.waitForNewFrame()
        color = frames["color"]
        depth = frames["depth"]
        color = color.asarray(np.uint8)
        depth = depth.asarray(np.float32)
        self.listener.release(frames)
        return color, depth
