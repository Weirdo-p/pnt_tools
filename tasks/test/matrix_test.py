path = "/home/xuzhuo/Documents/data/01-mini/20240717/proj/I300_GroundTruth_10hz.pos.bak"

import sys
import os, glob
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger


test = np.loadtxt(path)
test_rota = test[0, 4:].reshape(3, 3)
logger.info(test_rota.transpose() @ test_rota)