import numpy as np
import os
import sys


sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
# import utils.conversion.time as timeutil
from utils.rotation.dcm import Attitude2Azimuth

import argparse

parser = argparse.ArgumentParser(description='Convert attitude to azimuth')
parser.add_argument('-a', '--attitude', required=True, type=float, nargs=3, help='heading pitch roll')
parser.add_argument('--is_deg', type=bool, default=True, help='is degree')
args = parser.parse_args()

attitude = args.attitude
if args.is_deg:
    attitude = np.array(attitude) * np.pi / 180.0
azimuth = Attitude2Azimuth(attitude)
azimuth = azimuth * 180.0 / np.pi
logger.info("attitude: %s, azimuth: %s", attitude * 180.0 / np.pi, azimuth)
