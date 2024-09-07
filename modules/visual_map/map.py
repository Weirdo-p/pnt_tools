import numpy as np
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from modules.frame.frame import *

class MapPoint:
    def __init__(self, pos = np.zeros([3, 1]), mapPointId = -1):
        self.m_pos  = pos                       # p_{wj}^{w}
        self.m_id   = mapPointId                # ID              
        self.m_obs: list[Feature]  = []         # observations (features)

class Map:
    def __init__(self):
        self.m_points: dict[int: MapPoint] = {} # key: mappoint Id, value: mappoint
        self.m_frames: list[Frame] = []         # frames in window? or key frames

