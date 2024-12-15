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


root_path = "/home/xuzhuo/Documents/data/01-mini/20240702/cam00/lwh_features/Features"
frames = os.listdir(root_path)
frames = sorted(frames, key=lambda x: float(x[: -4]))

mapped_points = {}
for frame in frames:
    frame_path = os.path.join(root_path, frame)
    first_line = True
    with open(frame_path) as f:
        line = f.readline()
        while line:
            items = line.strip().split(" ")
            if first_line:
                frame_id, time = int(items[0]), float(items[1])
                if time < 2321 * 7 * 24 * 3600 + 179740:
                    break
                if time > 2321 * 7 * 24 * 3600 + 179741:
                    break
                first_line = False
                line = f.readline()
                continue
            if mapped_points.get(int(items[0])) is not None:
                mapped_points[int(items[0])].append(1)
            else:
                mapped_points[int(items[0])] = [1]
            line = f.readline()
            print(line)

logger.info(len(mapped_points))