import numpy as np
import enum
from enum import unique
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger

@unique
class CameraType(enum.Enum):
    STEREO = 0
    MONO   = 1

class Camera:
    def __init__(self, fx, fy, cx, cy, b = None, type = CameraType.MONO):
        self.m_fx                = fx
        self.m_fy                = fy
        self.m_cx                = cx
        self.m_cy                = cy
        self.m_b                 = b
        self.m_type              = type
        
    def project(self, xyz: np.ndarray) -> np.ndarray:
        """Project point in camera frame to image plane

        Args:
            xyz (ndarray): point in camera frame with shape(3, 1)
        """
        uv = np.zeros((3, 1))
        if (np.isnan(xyz[0, 0]) or np.isnan(xyz[1, 0]) or np.isnan(xyz[2, 0])):
            logger.critical(f"invalid mappoint {xyz.flatten()}")
            return None
        if self.m_type == CameraType.STEREO:
            uv[0, 0] = self.m_fx * xyz[0, 0] / (xyz[2, 0]) + self.m_cx
            uv[1, 0] = self.m_fy * xyz[1, 0] / (xyz[2, 0]) + self.m_cy
            uv[2, 0] = self.m_b / xyz[2, 0]
        elif self.m_type == CameraType.MONO:
            uv[0, 0] = self.m_fx * xyz[0, 0] / (xyz[2, 0]) + self.m_cx
            uv[1, 0] = self.m_fy * xyz[1, 0] / (xyz[2, 0]) + self.m_cy
            uv[2, 0] = 1
        else:
            logger.critical(f"Unsupported camera type: {self.m_type}")

        return uv

    def lift(self, uv: np.ndarray) -> np.ndarray:
        """Lift a pixel in image plane to camera frame

        Args:
            uv (ndarray): point in image plane with shape(3, 1)
        """
        xyz = np.zeros((3, 1))
        if self.m_type == CameraType.STEREO:
            z = self.m_b / uv[2]
            xyz[0, 0] = z * (uv[0, 0] - self.m_cx) / self.m_fx
            xyz[1, 0] = z * (uv[1, 0] - self.m_cy) / self.m_fy
            xyz[2, 0] = z
        elif self.m_type == CameraType.MONO:
            xyz[0, 0] = (uv[0, 0] - self.m_cx) / self.m_fx
            xyz[1, 0] = (uv[1, 0] - self.m_cy) / self.m_fy
            xyz[2, 0] = 1

        return xyz

    def getBaseline(self):
        if self.m_type == CameraType.STEREO:
            return self.m_b / self.m_fx
        else:
            logger.critical("[getBaseline] Only stereo cameras can call this function")


if __name__ == "__main__":
    logger.info("test on camera")
    uv = np.array([255, 500, 1]).reshape(3, 1)
    fx = 7.188560000000e+02
    fy = 7.188560000000e+02
    cx = 6.071928000000e+02
    cy = 1.852157000000e+02
    cam = Camera(fx, fy, cx, cy, type=CameraType.MONO)

    logger.info(
        (cam.project(cam.lift(uv)) - uv).flatten()
    )


