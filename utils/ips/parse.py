import numpy as np
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger

IPSSINSSOL_HEAD = [" YMD", "HMS", " Week"," GPSTime"," X-ECEF"," Y-ECEF"," Z-ECEF"," Latitude_deg"," Latitude_min"," Latitude_sec"," Longitude_deg"," Longitude_min"," Longitude_sec"," Ellip-Hgt"," Guass-X"," Guass-Y"," BaseID"," BL-E"," BL-N"," BL-U",
               " SD-E"," SD-N"," SD-U"," SD-EN"," SD-EU"," SD-NU"," VX-ECEF"," VY-ECEF"," VZ-ECEF"," VE"," VN"," VU"," SD-VE"," SD-VN"," SD-VU"," SD-VEN"," SD-VEU",
               " SD-VNU"," Yaw"," Pitch"," Roll"," SD-Yaw"," SD-Pitch"," SD-Roll"," SD-YP"," SD-YR"," SD-PR"," AccBiasX"," AccBiasY"," AccBiasZ"," SD-AccBiasX",
               " SD-AccBiasY"," SD-AccBiasZ"," GyrBiasX"," GyrBiasY"," GyrBiasZ"," SD-GyrBiasX"," SD-GyrBiasY"," SD-GyrBiasZ"," GPS(dT)"," GLO(dT)"," BD2(dT)",
               " BD3(dT)"," GAL(dT)"," QZS(dT)"," SD-GPSdT"," SD-GLOdT"," SD-BD2dT"," SD-BD3dT"," SD-GALdT"," SD-QZSdT"," ZTD"," ZWD"," SD-ZWD"," NSAT",
               " NSAT_AR"," NGPS_1"," NGPS_2"," NGPS_3"," NGLO_1"," NGLO_2"," NGLO_3"," NBD2_1"," NBD2_2"," NBD2_3"," NBD3_1"," NBD3_2"," NBD3_3"," NGAL_1",
               " NGAL_2"," NGAL_3"," NQZS_1"," NQZS_2"," NQZS_3"," HDOP"," VDOP"," Q"," ST"," AR"," Ratio"," RMS-L"," RMS-P"," Sig0-L"," Sig0-P",
               " IMUFlag"," Res-X"," Res-Y"," Res-Z"," Sep-E"," Sep-N"," Sep-U"," Sep-VE"," Sep-VN"," Sep-VU"," Sep-Yaw"," Sep-Pitch"," Sep-Roll"]

IPSGNSSSOL_HEAD = [" YMD", "HMS", "Week", "GPSTime", "X-ECEF", "Y-ECEF", "Z-ECEF", " Latitude_deg"," Latitude_min"," Latitude_sec"," Longitude_deg"," Longitude_min"," Longitude_sec", "Ellip-Hgt", "Guass-X", "Guass-Y", "BaseID", 
                   "BL-E", "BL-N", "BL-U", "SD-E", "SD-N", "SD-U", "SD-EN", "SD-EU", "SD-NU", "VX-ECEF", "VY-ECEF", "VZ-ECEF", "VE", "VN", "VU", 
                   "SD-VE", "SD-VN", "SD-VU", "SD-VEN", "SD-VEU", "SD-VNU", "Yaw", "Pitch", "Roll", "SD-Yaw", "SD-Pitch", "SD-Roll", "SD-YP", "SD-YR",
                   "SD-PR", "GPS(dT)", "GLO(dT)", "BD2(dT)", "BD3(dT)", "GAL(dT)", "QZS(dT)", "SD-GPSdT", "SD-GLOdT", "SD-BD2dT", "SD-BD3dT", "SD-GALdT",
                   "SD-QZSdT", "ZTD", "ZWD", "SD-ZWD", "NSAT", "NSAT_AR", " NGPS_1"," NGPS_2"," NGPS_3"," NGLO_1"," NGLO_2"," NGLO_3"," NBD2_1"," NBD2_2",
                   " NBD2_3"," NBD3_1"," NBD3_2"," NBD3_3"," NGAL_1", " NGAL_2"," NGAL_3"," NQZS_1"," NQZS_2"," NQZS_3", "HDOP", "VDOP", "Q", "ST", "AR",
                   "Ratio", "RMS-L", "RMS-P", "Sig0-L", "Sig0-P", "Est-Ref", "Est-Ref", "Est-Ref", "Sep-E", "Sep-N", "Sep-U"]
COMMENT = "#"
TYPE = str


class ParseIPSSol:
    def __init__(self) -> None:
        self.__IPSHead = list(np.char.strip(IPSSINSSOL_HEAD))
        self.__sol     = {}
        pass

    def parseSINS(self, path):
        self.__IPSHead = list(np.char.strip(IPSSINSSOL_HEAD))
        sol_str = np.loadtxt(path, dtype=TYPE, comments='#')
        for i in range(len(self.__IPSHead)):
            if i < 2:
                self.__sol[self.__IPSHead[i]] = sol_str[:, i]
                continue
            self.__sol[self.__IPSHead[i]] = sol_str[:, i].astype(np.float64)

        return self.__IPSHead, self.__sol
    
    def parseGNSS(self, path):
        self.__IPSHead = list(np.char.strip(IPSGNSSSOL_HEAD))
        sol_str = np.loadtxt(path, dtype=TYPE)
        print(sol_str.shape)
        print(len(IPSGNSSSOL_HEAD))

        for i in range(len(self.__IPSHead)):
            if i < 2:
                self.__sol[self.__IPSHead[i]] = sol_str[:, i]
                continue
            self.__sol[self.__IPSHead[i]] = sol_str[:, i].astype(np.float)

        return self.__IPSHead, self.__sol

    
    def getSol(self):
        self.__IPSHead = list(np.char.strip(IPSGNSSSOL_HEAD))
        return self.__IPSHead, self.__sol

