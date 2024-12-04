import rosbag
import glob
import matplotlib.pyplot as plt
import numpy as  np
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import cv2 as cv
import argparse
import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../")

from utils.logger.logger import logger

parser = argparse.ArgumentParser(description="Convert compressed image message to raw message (batch)")
parser.add_argument("bag_path", help="path to rosbag files", type=str)
parser.add_argument("save_path", help="Path to output", type=str)
args = parser.parse_args()


bridge = CvBridge()
bagpath = args.bag_path   # "2023-06-04-02-21-24_0.bag"
outpath = args.save_path

if not os.path.exists(outpath):
    logger.warning("output path do not exists, try to create")
    os.makedirs(outpath)
# os.
baglist = glob.glob(bagpath + "*.bag")
for each_bag in baglist:
    newbagpath = outpath + each_bag.split("/")[-1]
    bag = rosbag.Bag(each_bag, "r")
    new_bag = rosbag.Bag(newbagpath, "w")
    bag_data = bag.read_messages()
    logger.info("process {0}, save to {1}".format(each_bag, newbagpath))

    for topic, msg, t in bag_data:
        if "cam" not in topic or "compressed" not in topic:
            logger.info("Camera topic or compressed image not find")
            continue
        cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        raw_msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        raw_msg.header = msg.header
        items = topic.split("/")
        new_topic = "/{0}/{1}".format(items[1], items[2]) #items
        new_bag.write(new_topic, raw_msg, t)
    bag.close()
    new_bag.close()
