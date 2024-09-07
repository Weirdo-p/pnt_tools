import numpy as np
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from modules.frame.frame import *
from utils.matrix.matrix import *
from utils.rotation.dcm import *


class MapPoint:
    def __init__(self, pos = np.zeros([3, 1]), mapPointId = -1):
        self.m_pos  = pos                       # p_{wj}^{w}
        self.m_id   = mapPointId                # ID              
        self.m_obs: list[Feature]  = []         # observations (features)
    
    def triangulate(self):
        anchor_index = len(self.m_obs) - 1
        R_G2A, p_AinG = self.m_obs[anchor_index].m_frame.m_rota, self.m_obs[anchor_index].m_frame.m_pos
        R_AtoG = R_G2A.transpose()

        A, B = np.zeros((3, 3)), np.zeros((3, 1))

        for obs in self.m_obs:
            R_G2C, p_CinG = obs.m_frame.m_rota, obs.m_frame.m_pos
            R_A2C = R_G2C @ R_AtoG
            p_cinA = R_G2A @ (p_CinG - p_AinG)
            R_C2A = R_A2C.transpose()
            b = obs.m_PosInCamera
            b_i = R_C2A @ b
            b_i = 1.0 / np.linalg.norm(b_i, 2) * b_i
            b_i_skew = skew(b_i)
            Ai = b_i_skew.transpose() @ b_i_skew
            b_i_f = Ai @ p_cinA
            A += Ai
            B += b_i_f
        
        pos_anchor = np.linalg.inv(A) @ B
        return R_AtoG @ pos_anchor + p_AinG


	# double* R_G2A = Rec[Rec.size() - 1], *p_AinG = r_cam[Rec.size() - 1];
	# double R_AtoG[9] = {0};
	# MatrixTranspose(3, 3, R_G2A, R_AtoG);

	# double A[9] = {0}, B[3] = {0};

	# for (int i = 0; i < r_cam.size(); ++ i) {
	# 	double* R_G2C = Rec[i], *p_CinG = r_cam[i], *b = features[i];
	# 	double R_A2C[9] = {0}, R_C2A[9] = {0};
	# 	MatrixMultiply(3, 3, R_G2C, 3, 3, R_AtoG, R_A2C);
	# 	double p_cinA[3] = {0}, temp[3] = {0};
	# 	MatrixAddition(3, 1, p_CinG, p_AinG, temp, 1, -1);
	# 	MatrixMultiply(3, 3, R_G2A, 3, 1, temp, p_cinA);

	# 	MatrixTranspose(3, 3, R_A2C, R_C2A);
	# 	double b_i[3] = {0}, b_i_skew[9] = {0}, b_i_skew_T[9] = {0};
	# 	MatrixMultiply(3, 3, R_C2A, 3, 1, b, b_i);
	# 	MatrixScaleMultiply(1.0 / MatrixNorm2(3, 1, b_i), 3, 1, b_i);
	# 	MatrixSkewSymmetric(b_i, b_i_skew);

	# 	double Ai[9] = {0}, b_i_f[3] = {0};
	# 	MatrixTranspose(3, 3, b_i_skew, b_i_skew_T);
	# 	MatrixMultiply(3, 3, b_i_skew_T, 3, 3, b_i_skew, Ai);
	# 	MatrixMultiply(3, 3, Ai, 3, 1, p_cinA, b_i_f);

	# 	MatrixAddition2(3, 3, Ai, A);
	# 	MatrixAddition2(3, 1, b_i_f, B);
	# }

	# // MatrixWrite(stdout, 3, 3, A);
	# // MatrixWrite(stdout, 3, 1, B);

	# double A_inv[9] = {0};
	# MatrixInv(3, 3, A, A_inv);
	# double local[3] = {0};
	# MatrixMultiply(3, 3, A_inv, 3, 1, B, local);

	# MatrixMultiply(3, 3, R_AtoG, 3, 1, local, mapPoint);
	# MatrixAddition2(3, 1, p_AinG, mapPoint);











class Map:
    def __init__(self):
        self.m_points: dict[int: MapPoint] = {} # key: mappoint Id, value: mappoint
        self.m_frames: list[Frame] = []         # frames in window? or key frames

if __name__ == "__main__":
    MapPoint()
    Map()

