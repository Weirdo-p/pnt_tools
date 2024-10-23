import rosbag
import glob
import matplotlib.pyplot as plt
import numpy as  np
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import cv2 as cv
import argparse
import os
from utils.logger.logger import logger

# bagpath = ["/home/xuzhuo/Documents/data/01-mini/LVI-SAM-test/shanda/Camera2"]