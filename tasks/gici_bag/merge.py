import numpy as np
import os
import sys
import rosbag
from sensor_msgs.msg import Imu
import copy
import rospy


sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
import utils.conversion.time as timeutil


baglist = ["/media/xuzhuo/T7/01-data/09-gici/3.1/bag/image.bag",
           "/media/xuzhuo/T7/01-data/09-gici/3.1/bag/imu.bag"]

outbag = "/media/xuzhuo/T7/01-data/09-gici/3.1/bag/merge.bag"

outbag_data = rosbag.Bag(outbag, "w")
# outbag_data.write()

for each_bag in baglist:
    bagdata = rosbag.Bag(each_bag, "r")
    data = bagdata.read_messages()
    for topic, msg, t in data:
        week, sow = timeutil.UnixTime().toGPSWeek(msg.header.stamp.to_sec())
        msg.header.stamp = rospy.Time.from_sec(week * 7 * 24 * 3600 + sow)

        if "imu" in topic:

            new_imu = copy.deepcopy(msg)
            new_imu.angular_velocity.x = msg.angular_velocity.y
            new_imu.angular_velocity.y = msg.angular_velocity.z
            new_imu.angular_velocity.z = msg.angular_velocity.x
            new_imu.linear_acceleration.x = msg.linear_acceleration.y
            new_imu.linear_acceleration.y = msg.linear_acceleration.z
            new_imu.linear_acceleration.z = msg.linear_acceleration.x
            # msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
            outbag_data.write(topic, new_imu, msg.header.stamp)
            continue
        outbag_data.write(topic, msg, msg.header.stamp)
        
    bagdata.close()

outbag_data.close()
# data = bagdata.read_messages()
