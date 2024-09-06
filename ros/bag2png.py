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

parser = argparse.ArgumentParser(description="Extract pictures from specific rosbags and topics")
parser.add_argument("bag_path", help="path to rosbag file", type=str)
parser.add_argument("topic", help="topic name", type=str)
parser.add_argument("save_path", help="Path to output pictures with topic name", type=str)
args = parser.parse_args()

bagfile = args.bag_path
save_path = args.save_path
topic_require = args.topic

if not os.path.exists(save_path):
    logger.error("given save path not exist")
    exit(0)

if not os.path.exists(bagfile):
    logger.error("given bagfile not exist")
    exit(0)

save_path = save_path + "/" + topic_require + "/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger.info("image will be saved at {0}".format(save_path))
bagdata = rosbag.Bag(bagfile, "r")
data = bagdata.read_messages()

bridge = CvBridge()
for topic, msg, t in data:
    if topic == topic_require:
        cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')
        timestr = "{0:6f}.png".format(msg.header.stamp.to_sec())
        cv.imwrite(save_path+timestr, cv_image)

bagdata.close()
