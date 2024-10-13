import rosbag
import numpy as  np
import argparse
import os, glob, sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../")

from utils.logger.logger import logger
import utils.conversion.time as timeutil

Rad2Deg = 180 / np.pi

bag_name = "/media/xuzhuo/T7/01-data/09-gici/3.1/bag/imu.bag"
imu_file = "/media/xuzhuo/T7/01-data/09-gici/3.1/bag/imu.imu"
bagdata = rosbag.Bag(bag_name, "r")
data = bagdata.read_messages()



# bridge = CvBridge()
with open(imu_file, "w") as f:
    for topic, msg, t in data:
        week, sow = timeutil.UnixTime().toGPSWeek(msg.header.stamp.to_sec())
        f.write(f"{sow} {msg.angular_velocity.x} {msg.angular_velocity.y} {msg.angular_velocity.z} {msg.linear_acceleration.x} {msg.linear_acceleration.y} {msg.linear_acceleration.z}\n")
        # print(msg)
        # print(week, sow)
        # pass