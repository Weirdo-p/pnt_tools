import numpy as np
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger


URF2RFU = np.array([[0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])

RCtoURF = np.array([[0.0134381210697455122,   -0.999901594987671172, -0.00402706384669154413,  -0.0229793554058591656],
                [0.999907681540912807,   0.0134460859659659704, -0.00195733688253543802,   0.0110786309679912626],
                [0.00201129251744842832, -0.00400038914436078377,     0.99998997574430859,    0.025008868367930974],
                [                    0,                       0,                       0,                       1]])
RCtoRFU = URF2RFU @ RCtoURF

print(RCtoRFU)