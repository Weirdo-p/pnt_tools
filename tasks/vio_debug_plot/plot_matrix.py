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
from utils.ips import parse

path_sol = "/home/xuzhuo/Documents/code/C++/IPS-CMake-BuildProject-Linux/log/debug.txt"

color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
        2: [(255 / 255), (102 / 255), (102 / 255)],  # red
        1: [(255 / 255), (146 / 255), (0 / 255)],  # blue
        0: [(0 / 255), (141 / 255), (0 / 255)],
        4: [(20 / 255), (169 / 255), (89 / 255)],
        5: [(70 / 255), (114 / 255), (196 / 255)],
        6: [(150 / 255), (150 / 255), (150 / 255)]}  # green
data = np.loadtxt(path_sol)
flag = (data != 0)
logger.info(flag)
data[flag] = 1
logger.info(data)
np.savetxt("/home/xuzhuo/Documents/code/python/smartpnt_tools/log/test.txt", data)
plt.figure()
plt.matshow(data)
plt.show()