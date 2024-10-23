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
#%% mappoint id check
file_name = "/home/xuzhuo/Documents/code/C++/IPS-CMake-BuildProject-Linux/log/cov_debug1.txt"
depth_data = np.loadtxt(file_name)
cov = depth_data
mappoint_freq = {}
# for i in range(point_id.shape[0]):
#     mappoint_freq[point_id[i]] = mappoint_freq[point_id[i]] + 1 if mappoint_freq.get(point_id[i]) is not None else 1

print(mappoint_freq)
plt.scatter(range(len(cov)), list(cov))
plt.show()