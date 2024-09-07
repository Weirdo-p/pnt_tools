import numpy as np
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from modules.visual_map.map import *
from modules.camera.camera import *

class Frame:
    def __init__(self, id = 0, pos = np.zeros([3, 1]), rotation = np.identity(3)):
        self.m_pos       =  pos                  # position p_{wc}^{w}
        self.m_rota      =  rotation             # rotation R_{w}^{c}
        self.m_id        =  id
        self.m_features: list[Feature]  =  []    # features
        self.m_time      = 0

    def project_to_camera(self, mp: MapPoint):
        return self.m_rota @ (mp.m_pos - self.m_pos)

    def project_to_image(self, mp:MapPoint, camera: Camera):
        cam_pos = self.project_to_camera(mp)
        uv = camera.project(cam_pos)
        if not camera.in_image(uv): 
            logger.warning(f"{uv.flatten()} is not on the image. image resolution: {camera.m_resolution}")
        return uv


class Feature:
    def __init__(self, pos = np.zeros([3, 1]), du = 0, mapPointId = -1):
        self.m_pos = pos                        # pixels
        self.m_PosInCamera = np.zeros([3, 1])
        self.m_du = du                          # if it is stereo cameras
        self.m_mapPointId = mapPointId          # 该像素对应的地图点（三维点）Id
        self.m_mappoint = None
        self.m_frame: Frame = None

