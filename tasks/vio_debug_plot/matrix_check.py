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

#%%
path_1 = "/home/xuzhuo/Documents/code/C++/IPS-CMake-BuildProject-Linux/log/compare/GI/debug.txt"
path_2 = "/home/xuzhuo/Documents/code/C++/IPS-CMake-BuildProject-Linux/log/compare/GVIO/debug.txt"

matrix1 = np.loadtxt(path_1)
matrix2 = np.loadtxt(path_2)

diff = matrix1 - matrix2
np.savetxt("./diff.txt", diff)