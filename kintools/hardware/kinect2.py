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
import cv2
from kintools.hardware import utils


class Kinect2:
    def __init__(self, calibration_folder=None):
        self.freenect = Freenect2()
        setGlobalLogger(createConsoleLogger(LoggerLevel.NONE))
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

        self.calibration = {}
        if calibration_folder is not None:
            self.calibration = utils.load_calibration(calibration_folder)

    def snapshot(self):
        frames = self.listener.waitForNewFrame()
        color = frames["color"]
        depth = frames["depth"]
        ir = frames["ir"]
        color = color.asarray(np.uint8).copy()
        depth = depth.asarray(np.float32).copy()
        ir = ir.asarray(np.float32).copy()

        # flip images
        color = color[:, ::-1]
        ir = ir[:, ::-1]
        depth = depth[:, ::-1]

        color = color[..., :3]
        # color = color[..., ::-1]
        ir = self._convert_ir(ir)

        self.listener.release(frames)
        return color, depth, ir

    def _convert_ir(self, ir):
        """
        convertIr function
        https://github.com/code-iai/iai_kinect2/blob/master/kinect2_calibration/src/kinect2_calibration.cpp
        """
        ir_min, ir_max = ir.min(), ir.max()
        ir_convert = (ir - ir_min) / (ir.max()-ir.min())
        ir_convert *= 255
        ir_convert = np.tile(ir_convert[..., None], (1, 1, 3))
        return ir_convert.astype(np.uint8)
