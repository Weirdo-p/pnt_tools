import numpy as np
import math
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
def BLH2NEU(station, obj):
    '''

    :param station: BLH in deg
    :param obj: BLH in deg
    :return:
    '''
    # print(station, obj)
    rotation = np.zeros([3, 3])
    B, L = Deg2Rad(station[0]), Deg2Rad(station[1])
    rotation[0, 0] = -np.sin(B) * np.cos(L)
    rotation[0, 1] = -np.sin(B) * np.sin(L)
    rotation[0, 2] = np.cos(B)

    rotation[1, 0] = -np.sin(L)
    rotation[1, 1] = np.cos(L)
    rotation[1, 2] = 0

    rotation[2, 0] = np.cos(B) * np.cos(L)
    rotation[2, 1] = np.cos(B) * np.sin(L)
    rotation[2, 2] = np.sin(B)

    obj_xyz = BLH2XYZ(obj)
    station_xyz = BLH2XYZ(station)
    vec = obj_xyz - station_xyz
    return np.matmul(rotation, vec)



def XYZ2BLH(xyz):
    e2_ = 0.0066943800042608276
    a_ = 6378137
    blh = np.zeros([3,1])

    blh[1] = Rad2Deg(math.atan2(xyz[1], xyz[0]))
    iteration = 0
    B0 = 5
    while(iteration != 10):
        # print (B0)
        sinB = math.sin(Deg2Rad(B0))
        W = np.sqrt(1 - e2_ * sinB * sinB)
        N = a_ / W
        H = xyz[2] / np.sin(Deg2Rad(B0)) - N * (1 - e2_)
        up = xyz[2] + N * e2_ * sinB
        down = np.sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1])
        blh[0] = math.atan(up / down)
        blh[0] = Rad2Deg(blh[0])
        blh[2] = H
        # error = abs(blh[0] - B0)
        # if(error < 1e-20):
        #     break
        B0 = blh[0]
        iteration += 1

    # blh[0] = Deg2Rad(blh[0])
    # blh[1] = Deg2Rad(blh[1])
    # print (blh)
    return blh

def XYZ2NEU(station, obj):
    blh_station = XYZ2BLH(station)
    blh_obj = XYZ2BLH(obj)
    neu = BLH2NEU(blh_station, blh_obj)
    return neu

def BLH2XYZ(blh):
    '''

    :param blh: in deg
    :return:
    '''
    xyz = np.zeros([3])
    e2_ = 0.0066943800042608276
    a_ = 6378137
    B_rad = Deg2Rad(blh[0])
    L_rad = Deg2Rad(blh[1])
    sinB = np.sin(B_rad)
    cosB = np.cos(B_rad)
    sinL = np.sin(L_rad)
    cosL = np.cos(L_rad)

    sub = np.sqrt(1 - e2_ * sinB * sinB)
    N = a_ / sub

    xyz[0] = (N + blh[2]) * cosB * cosL
    xyz[1] = (N + blh[2]) * cosB * sinL
    xyz[2] = (N * (1 - e2_) + blh[2]) * sinB
    # print(xyz)
    return xyz


def Deg2Rad(deg):
    return deg / 180 * np.pi

def Rad2Deg(rad):
    return rad / np.pi * 180