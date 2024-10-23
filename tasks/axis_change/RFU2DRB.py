import numpy as np
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger


RFU2DRB = np.array([[0, 0, -1, 0],
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]])

RCtoURF = np.array([[ 0.99904543, -0.01375358,  0.04146159,  -0.07527576],
                    [-0.04119367,  0.0192227 ,  0.99896625,  0.18479526],
                    [-0.01453636, -0.99972062,  0.01863779,    -0.24771659],
                    [                    0,                       0,                       0,                       1]])
RCtoRFU = RFU2DRB @ RCtoURF

print(RCtoRFU)