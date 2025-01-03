import numpy as np
import math
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger


EPSILON = 2.2204460492503131e-016

def Angle2RotMatrix(angle:np.array) -> np.array:
    """Convert Euler angles to Rotation matrix

    Args:
        angle (np.array): Euler angle in radians(yaw pitch roll)

    Returns:
        np.array: Directional Cosine Matrix
    """
    ca = math.cos(angle[0])
    sa = math.sin(angle[0])
    cr = math.cos(angle[1])
    sr = math.sin(angle[1])
    cb = math.cos(angle[2])
    sb = math.sin(angle[2])


    R = np.array(
        [[ca*cb - sa*sb*sr, sa*cb + ca*sb*sr, -sb*cr],
         [-sa*cr, ca*cr, sr],
         [ca*sb + sa*cb*sr, sa*sb - ca*cb*sr, cb*cr]]
    )

    return R


def RotMatrix2RotAngle(rot_mat: np.array) -> np.array:
    angle = np.zeros((3, 1))
    if abs(rot_mat[0, 2]) < EPSILON and abs(rot_mat[2, 2]) < EPSILON:
        logger.error("RotaModel is Z-X-Y, and the pitch avoid ±90!")

        angle[0] = 0.0
        angle[1] = math.atan(rot_mat[1, 2])
        angle[2] = math.atan2(rot_mat[2, 0], rot_mat[0, 0])
    else:
        angle[0] = math.atan2(-rot_mat[1, 0], rot_mat[1, 1])
        angle[1] = math.asin(rot_mat[1, 2])
        angle[2] = math.atan2(-rot_mat[0, 2], rot_mat[2, 2])

    return angle


def Azimuth2Attitude(azimuth):
    yaw = azimuth[0]

    if yaw <= np.pi:
        yaw = -yaw
    else:
        yaw = np.pi * 2 - yaw
    attitude = np.array([yaw, azimuth[1], azimuth[2]])
    return attitude


def Attitude2Azimuth(attitude):
    yaw = attitude[0]
    
    if yaw < 0.0:
        yaw = -yaw; 
    else:
        yaw = np.pi * 2 - yaw

    azimuth = np.array([yaw, attitude[1], attitude[2]])

    return azimuth


if __name__ == "__main__":
    angle = np.array([30, 79, 26]) * np.pi / 180.0
    logger.info("test on rotation conversion")
    logger.info("differences: ")
    logger.info(
            (RotMatrix2RotAngle(Angle2RotMatrix(angle)) - angle.reshape(3, 1)).flatten()
    )
    

