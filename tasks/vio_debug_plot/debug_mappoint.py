# %% import libraries
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


file = "/home/xuzhuo/Documents/code/C++/IPS-CMake-BuildProject-Linux/log/mappoint_debug.txt"
mappoints = np.loadtxt(file)

point_freq = {}
for i in range(mappoints.shape[0]):
    if point_freq.get(mappoints[i, 0]) is not None:
        # print(point_freq.get(mappoints[i, 0]))
        point_freq[mappoints[i, 0]].append(mappoints[i, 1])
    else:  point_freq[mappoints[i, 0]] = [mappoints[i, 1]]

# print(point_freq)
plt.figure()
for k, v in point_freq.items():
    plt.plot(range(len(v)), v)
    plt.show()