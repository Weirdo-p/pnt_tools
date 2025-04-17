import numpy as np
import sys
import os
from scipy import spatial
import matplotlib.pyplot as plt

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from utils.ie.parse_gici import ParseIESol
from utils.ips.parse import ParseIPSSol
from utils.conversion.coordinate import BLH2XYZ, Attitude2Azimuth, Deg2Rad, XYZ2NEU, Angle2RotMatrix, Attitude2Rbe

BASE_STATION = np.array([-2853687.53199,  4665985.97952,  3270117.40020])

def CheckSINS():
    path = "/home/xuzhuo/Documents/data/gici/new/1.1/ground_truth.txt"
    head_gt, gt = ParseIESol().parseSINS(path)
    logger.info(gt)
    gt_tree = spatial.KDTree(np.array(gt["GPSTime"]).reshape(-1, 1))

    path = "/home/xuzhuo/Documents/data/gici/new/5.2/bag/gnss_rover(2).flf"
    head, sol = ParseIPSSol().parseSINS(path)
    logger.info(sol["GPSTime"])
    pos_err, att_err, valid_time = [], [], []


    for idx, time in enumerate(sol["GPSTime"]):
        if time % 1 != 0: continue

        result = gt_tree.query(time)

        if result[0] != 0: continue
        
        pos = np.array([sol["X-ECEF"][idx], sol["Y-ECEF"][idx], sol["Z-ECEF"][idx]])
        pos_enu = XYZ2NEU(BASE_STATION, pos)
        att = np.array([sol["Yaw"][idx], sol["Pitch"][idx], sol["Roll"][idx]])

        pos_gt = BLH2XYZ(np.array([Deg2Rad(gt["Latitude"][result[1]]), Deg2Rad(gt["Longitude"][result[1]]), gt["H-Ell"][result[1]]]))
        pos_gt_enu = XYZ2NEU(BASE_STATION, pos_gt)
        att_gt = Attitude2Azimuth(np.array([gt["Heading"][result[1]], gt["Pitch"][result[1]], gt["Roll"][result[1]]]))

        pos_err.append(pos_enu - pos_gt_enu)
        att_err.append(att - att_gt)
        valid_time.append(time)
        # valid_num += 1
        # logger.info("test")
    
    plt.figure(0)
    pos_err = np.array(pos_err)
    plt.plot(valid_time, pos_err[:, 0], label = "N")
    plt.plot(valid_time, pos_err[:, 1], label = "E")
    plt.plot(valid_time, pos_err[:, 2], label = "U")
    plt.legend()
    plt.show()

    plt.figure(1)
    att_err = np.array(att_err)
    plt.plot(valid_time, att_err[:, 0], label = "Yaw")
    plt.plot(valid_time, att_err[:, 1], label = "Pitch")
    plt.plot(valid_time, att_err[:, 2], label = "Roll")
    plt.legend()
    plt.show()


def CheckGNSS():
    path = "/home/xuzhuo/Documents/data/gici/new/5.2/ground_truth.txt.nmea.transformed.translated.ie"
    head_gt, gt = ParseIESol().parseSINS(path)
    logger.info(gt)
    gt_tree = spatial.KDTree(np.array(gt["GPSTime"]).reshape(-1, 1))

    path = "/home/xuzhuo/Documents/data/gici/new/5.2/IPSProj/R_RTK/gnss_rover.ffp"
    head, sol = ParseIPSSol().parseGNSS(path)
    logger.info(sol["GPSTime"])
    pos_err, att_err, valid_time = [], [], []


    for idx, time in enumerate(sol["GPSTime"]):
        if time % 1 != 0: continue

        result = gt_tree.query(time)

        if result[0] != 0: continue
        
        pos = np.array([sol["X-ECEF"][idx], sol["Y-ECEF"][idx], sol["Z-ECEF"][idx]])
        pos_enu = XYZ2NEU(BASE_STATION, pos)
        # att = np.array([sol["Yaw"][idx], sol["Pitch"][idx], sol["Roll"][idx]])

        pos_gt = BLH2XYZ(np.array([Deg2Rad(gt["Latitude"][result[1]]), Deg2Rad(gt["Longitude"][result[1]]), gt["H-Ell"][result[1]]]))
        pos_gt_enu = XYZ2NEU(BASE_STATION, pos_gt)
        # att_gt = Attitude2Azimuth(np.array([gt["Heading"][result[1]], gt["Pitch"][result[1]], gt["Roll"][result[1]]]))

        pos_err.append(pos_enu - pos_gt_enu)
        # att_err.append(att - att_gt)
        valid_time.append(time)
        # valid_num += 1
        # logger.info("test")
    
    plt.figure(0)
    pos_err = np.array(pos_err)
    plt.plot(valid_time, pos_err[:, 0], label = "N")
    plt.plot(valid_time, pos_err[:, 1], label = "E")
    plt.plot(valid_time, pos_err[:, 2], label = "U")
    plt.legend()
    plt.show()




def check_lever_arm():
    path = "/home/xuzhuo/Documents/data/gici/new/1.1/ground_truth.txt.nmea.transformed.ie"
    head_gt, gt = ParseIESol().parseSINS(path)
    logger.info(gt)
    gt_tree = spatial.KDTree(np.array(gt["GPSTime"]).reshape(-1, 1))

    path = "/home/xuzhuo/Documents/data/gici/new/1.1/IPSProj/R_RTK/gnss_rover.ffp"
    head, sol = ParseIPSSol().parseGNSS(path)
    logger.info(sol["GPSTime"])
    pos_err, att_err, valid_time = [], [], []


    for idx, time in enumerate(sol["GPSTime"]):
        if time % 1 != 0: continue

        result = gt_tree.query(time)

        if result[0] != 0: continue
        
        pos = np.array([sol["X-ECEF"][idx], sol["Y-ECEF"][idx], sol["Z-ECEF"][idx]])
        # pos_enu = XYZ2NEU(BASE_STATION, pos)


        pos_gt = BLH2XYZ(np.array([Deg2Rad(gt["Latitude"][result[1]]), Deg2Rad(gt["Longitude"][result[1]]), gt["H-Ell"][result[1]]]))
        att = np.array([Deg2Rad(gt["Heading"][result[1]]), Deg2Rad(gt["Pitch"][result[1]]), Deg2Rad(gt["Roll"][result[1]])])
        Rbe = Attitude2Rbe(att, pos_gt)
        # pos_gt_enu = XYZ2NEU(BASE_STATION, pos_gt)
        # att_gt = Attitude2Azimuth(np.array([gt["Heading"][result[1]], gt["Pitch"][result[1]], gt["Roll"][result[1]]]))

        pos_err.append(np.linalg.inv(Rbe)@(pos - pos_gt))
        # att_err.append(att - att_gt)
        valid_time.append(time)
    plt.figure(0)
    pos_err = np.array(pos_err)
    plt.plot(valid_time, pos_err[:, 0], label = "R")
    plt.plot(valid_time, pos_err[:, 1], label = "F")
    plt.plot(valid_time, pos_err[:, 2], label = "U")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # CheckGNSS()
    CheckSINS()

    # check_lever_arm()