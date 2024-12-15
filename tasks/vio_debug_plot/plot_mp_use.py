import numpy as np
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from modules.camera.camera import *
from modules.frame.frame import *
from modules.visual_map.map import *
from utils.rotation.dcm import *
import matplotlib.pyplot as plt

used_pts = "/home/xuzhuo/Documents/code/C++/IPS-CMake-BuildProject-Linux/log/mappoint_debug1.txt"
mps = np.loadtxt(used_pts)
plt.figure()
plt.plot(range(mps.shape[0]), mps[:, 0], color="r", linewidth=2.5)
plt.plot(range(mps.shape[0]), mps[:, 1], color="b", linewidth=2.5)
plt.show()