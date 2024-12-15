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

root_path = "/home/xuzhuo/FeatUpdate.txt"
used_pts = {}
with open(root_path) as f:
    line = f.readline().strip("\n")
    while line:
        items = line.split("  ")
        time = items[0]
        used_pts[time] = []
        for i in range(1, len(items)):
            used_pts[time].append(int(items[i]))
        line = f.readline()

logger.warning(len(used_pts["179740.600087"]))