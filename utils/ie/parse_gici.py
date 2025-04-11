import numpy as np
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger


# logger.info("Week     GPSTime     UTCTime       Longitude       Latitude     H-Ell         Heading           Pitch            Roll".split())

IE_SINS_HEAD = ['Week', 'GPSTime', 'UTCTime', 'Longitude', 'Latitude', 'H-Ell', 'Heading', 'Pitch', 'Roll']
TYPE = str

class ParseIESol:
    def __init__(self) -> None:
        self.__IPSHead = list(np.char.strip(IE_SINS_HEAD))
        self.__sol     = {}
        pass

    def parseSINS(self, path):
        self.__IPSHead = list(np.char.strip(IE_SINS_HEAD))
        sol_str = np.loadtxt(path, dtype=TYPE, comments='#')
        for i in range(len(self.__IPSHead)):
            if i < 2:
                self.__sol[self.__IPSHead[i]] = sol_str[:, i]
                continue
            self.__sol[self.__IPSHead[i]] = sol_str[:, i].astype(np.float64)

        return self.__IPSHead, self.__sol
    
if __name__ == "__main__":
    path = "/home/xuzhuo/Documents/data/gici/new/1.1/ground_truth.txt"
    head, sol = ParseIESol().parseSINS(path)
    logger.info(sol)