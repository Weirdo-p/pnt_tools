import numpy as np
import math
import copy

class Frame:
    def __init__(self, id = 0, pos = np.zeros([3, 1]), rotation = np.identity(3)):
        self.m_pos       =  pos                  # position p_{wc}^{w}
        self.m_rota      =  rotation             # rotation R_{w}^{c}
        self.m_id        =  id
        self.m_features: list[Feature]  =  []    # features
        self.m_time      = 0

class Feature:
    def __init__(self, pos = np.zeros([3, 1]), du = 0, mapPointId = -1):
        self.m_pos = pos                        # pixels
        self.m_PosInCamera = np.zeros([3, 1])
        self.m_du = du                          # if it is stereo cameras
        self.m_mapPointId = mapPointId          # 该像素对应的地图点（三维点）Id
        self.m_mappoint = None
        self.m_frame = None

