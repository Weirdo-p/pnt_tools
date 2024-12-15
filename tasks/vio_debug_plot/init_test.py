import numpy as np
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from utils.rotation.dcm import *
from modules.camera.camera import *
from modules.frame.frame import *
from modules.visual_map.map import *
from utils.rotation.dcm import *
from utils.conversion.coordinate import *
import matplotlib.pyplot as plt


rota_lwh = np.array([0.108108563, 0.927220464, -0.358573214, -0.475455887,  0.364985814,  0.800454280,  0.873071726,  0.083949784,  0.480310520])
rota_lwh = rota_lwh.reshape(3, 3)
pos_lwh = np.array([-2267666.071, 5009215.058, 3221241.593])
llh_lwh = XYZ2LLH(pos_lwh, 0)
logger.warning("azimuth from LWH")
logger.info(Attitude2Azimuth(Rbe2Attitude(rota_lwh, llh_lwh)).flatten() * 180 / np.pi)


rota_xz = np.array([0.17102986163981,  0.916516276224429, -0.361589134022544, -0.451090291860581,  0.399113139312678,  0.798264524213067,  0.875937403594917, 0.0265822768717144,  0.481691859532328])
pos_xz = np.array([-2267665.71534821, 5009216.52522527, 3221242.43941048])
rota_xz = rota_xz.reshape(3, 3)
llh_xz = XYZ2LLH(pos_xz, 0)
logger.warning("azimuth from XZ")
logger.info(Attitude2Azimuth(Rbe2Attitude(rota_xz, llh_xz)).flatten() * 180 / np.pi)

logger.warning("azimuth from gt")
logger.info([272.356931769, -0.0850899622, -0.4719625246])

llh_test = np.array([-2267665.87574416,  5009215.08312945,  3221241.59848352])
test = np.array([0.100078007903083,   0.92562169730339, -0.364977897707987,  -0.48145894162195,  0.366065516139238,  0.796362559030737,  0.870736286062791, 0.0960234938694944,  0.482283950343232]).reshape(3, 3)
logger.info(Attitude2Azimuth(Rbe2Attitude(test, llh_test)).flatten() * 180 / np.pi)

llh_test1 = np.array([-2267665.877,  5009215.081,  3221241.599])
test1 = np.array([0.100214005,  0.925609052, -0.364972651, -0.481410294,  0.366131171,  0.796361787,  0.870747542,  0.095894987,  0.482289196]).reshape(3, 3)
logger.info(Attitude2Azimuth(Rbe2Attitude(test, llh_test1)).flatten() * 180 / np.pi)
