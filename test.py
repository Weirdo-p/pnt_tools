from utils.logger.logger import logger
from utils.coors.rotation import *

if __name__ == "__main__":
    
    angle = [0, 0, 0]
    logger.info("test on rotation")
    logger.info((RotMatrix2RotAngle(Angle2RotMatrix(angle)) - np.array(angle).reshape(3, 1)).flatten())