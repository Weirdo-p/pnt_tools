path = "/home/xuzhuo/Documents/data/01-mini/20240717/proj/I300_GroundTruth_10hz.pos.bak"

import sys
import os, glob
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger


test = np.loadtxt(path)
test_rota = test[0, 4:].reshape(3, 3)
# logger.info(test_rota.transpose() @ test_rota)

A = np.array([[-0.999871192442087553,   -0.0128272212687904081,  0.009646808744800929036, -0.001631526123218773189],
        [-0.01281568650175066333,    0.9999170861579332881,  0.001256578257457377414,   -0.3204949121983169391],
        [-0.009662127298174172374,  0.001132785924064993229,   -0.9999526789264197024,   0.06657108425640941018],
        [                      0,                        0,                        0,                        1]])

logger.info(A)
logger.info(np.linalg.inv(A))