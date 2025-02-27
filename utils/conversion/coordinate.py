import numpy as np
import math
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from utils.rotation.dcm import *

D2R = (0.0174532925199432957692369076849)     # deg to rad
R2D = (57.295779513082322864647721871734)     # rad to deg

gs_WGS84_a = 6378137.0                                           # earth semimajor axis (WGS84) (m)
gs_WGS84_b = 6356752.31425                                       # earth semimajor axis (WGS84) (m)
gs_WGS84_FE = 1.0 / 298.257223563                                # earth flattening (WGS84)
gs_WGS84_e2 = 2 * gs_WGS84_FE - (gs_WGS84_FE ** 2)                 # 
gs_WGS84_e = np.sqrt(gs_WGS84_e2)                                   # earth first eccentricity
gs_WGS84_GME = 3.986004418E+14                                   # earth gravitational constant (m^3/s^2) fM
gs_WGS84_J2 = 1.08263E-03                                        # 
gs_WGS84_J4 = -2.37091222E-6                                     # 
gs_WGS84_J6 = 6.08347E-9                                         # 
gs_WGS84_OMGE = 7.2921151467E-5                                  # earth angular velocity (IS-GPS) (rad/s)
gs_WGS84_Ge = 9.7803267714                                       # gravity at equator (m/s^2) 
gs_WGS84_Gp = 9.8322011865                                       # gravity at polar   (m/s^2) 
gs_WGS84_Gm = 9.7976446561                                       # Mean value (normal) gravity (m/s^2)
gs_WGS84_Galph = 0.00193336138707                                # gravity formula constant
gs_CGCS2000_a = 6378137.0                                        # earth semimajor axis (WGS84) (m)
gs_CGCS2000_b = 6356752.314140356;                               # earth semimajor axis (WGS84) (m)
gs_CGCS2000_FE = 1.0 / 298.257222101                             # earth flattening (WGS84)
gs_CGCS2000_e2 = 2 * gs_CGCS2000_FE - (gs_CGCS2000_FE ** 2)
gs_CGCS2000_e = np.sqrt(gs_CGCS2000_e2)                             # earth first eccentricity
gs_CGCS2000_GME = 3.986004418E+14                                # earth gravitational constant (m^3/s^2) fM
gs_CGCS2000_OMGE = 7.2921150E-5                                  # earth angular velocity (IS-GPS) (rad/s)
gs_min = D2R / 60.0                                              # 一个单位角分 对应的 弧度
gs_sec = D2R / 3600.0                                            # 一个单位角秒 对应的 弧度
gs_dps = D2R                                                     # 1度/秒 对应的 弧度/秒
gs_dph = D2R / 3600.0                                            # 1度/小时 对应的 弧度/秒
gs_dpss = D2R                                                    # 1度/xsqrt(秒) 对应的 弧度/xsqrt(秒)
gs_dpsh = D2R / 60.0                                             # 1度/xsqrt(小时) 对应的 弧度/xsqrt(秒), 角度随机游走系数
gs_g0 = gs_WGS84_Gm                                              # 重力加速度(米/秒平方)
gs_ug = gs_g0 * 1e-6                                             # 毫重力加速度
gs_mg = gs_g0 * 1e-3                                             # 微重力加速度
gs_ppm = 1e-6                                                    # part per million(百万分之一)

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

    blh[0] = Deg2Rad(blh[0])
    blh[1] = Deg2Rad(blh[1])
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



# /**
# * @brief       WGS84坐标系: XYZ坐标转LLH坐标
# * @param[in]   XYZ          double   XYZ(m) 坐标
# * @param[in]   CoorSys      int      坐标系统: 0:WGS84, 1:CGCS2000
# * @param[out]  LLH          double   Lat[-90,90],Lon[-180,180],Hgt (deg,m)
# * @return      void
# * @note
# * @par History:
# *              2018/01/17,Feng Zhu, new \n
# * @internals
# */
def XYZ2LLH(XYZ, CoorSys):
    LLH = np.zeros(3)
    a = gs_WGS84_a
    e2 = gs_WGS84_e2

    if (CoorSys == 1):
        a = gs_CGCS2000_a
        e2 = gs_CGCS2000_e2

    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    r2 = X * X + Y * Y
    z = 0.0
    zk = 0.0
    v = a
    sinp = 0.0

    z, zk = Z, 0.0
    while np.abs(z - zk) >= 1E-4:
        zk = z
        sinp = z / np.sqrt(r2 + z * z)
        v = a / np.sqrt(1.0 - e2 * sinp*sinp)
        z = Z + v * e2*sinp
    
    if r2 > 1E-12:
        LLH[0] = np.arctan(z / np.sqrt(r2))
        LLH[1] = np.arctan2(Y, X)
    else:
        LLH[1] = 0
        if (Z > 0.0): LLH[0] = np.pi / 2.0
        else: LLH[0] = -np.pi / 2.0
    
    LLH[2] = np.sqrt(r2 + z * z) - v
    return LLH


# /**
# * @brief       E系下的旋转矩阵(Reb)转换到L系下的旋转矩阵(Rlb)
# * @param[in]   E_R       double     E系下的旋转矩阵(Reb)
# * @param[in]   LLH       double     纬度,经度,大地高
# * @param[out]  L_R       double     L系下的旋转矩阵(Rlb)
# * @return
# * @note
# * @par History:
# *              2018/01/17,Feng Zhu, new \n
# * @internals
# */
def Rbe2Rbn(Rbe:np.ndarray, LLH:np.ndarray):
    Reb = Rbe.transpose()
    Rne = Renu2xyz(LLH)
    return (Reb @ Rne).transpose()

# /**
# * @brief       ENU转向ECEF系的旋转矩阵
# * @param[in]   LLH       double     纬度,经度,大地高
# * @param[out]  R_L2E     double     ENU->ECEF的旋转矩阵
# * @return
# * @note
# * @par History:
# *              2018/01/17,Feng Zhu, new \n
# * @internals
# */
def Renu2xyz(LLH:np.ndarray):
    Rne = np.zeros((3, 3))
    sinp = np.sin(LLH[0])
    cosp = np.cos(LLH[0])
    sinl = np.sin(LLH[1])
    cosl = np.cos(LLH[1])
    Rne[0, 0]=-sinl 
    Rne[0, 1]=-sinp*cosl
    Rne[0, 2]=cosp*cosl
    Rne[1, 0]=cosl
    Rne[1, 1]=-sinp*sinl
    Rne[1, 2]=cosp*sinl
    Rne[2, 0]=0.0
    Rne[2, 1]=cosp
    Rne[2, 2]=sinp
    return Rne

def Rbe2Attitude(Rbe, LLH):
    # double Rlb[9]
    # double Reb[9]
    # MatrixTranspose(3, 3, Rbe, Reb);
    Rbn = Rbe2Rbn(Rbe, LLH)
    
    return RotMatrix2RotAngle(Rbn.transpose())

def Attitude2Rbe(att, xyz):
    llh = XYZ2BLH(xyz)
    Rne = Renu2xyz(llh).transpose()
    Rnb = Angle2RotMatrix(att)

    return (Rnb @ Rne).transpose()


