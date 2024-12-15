# %% import libraries
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from modules.camera.camera import *
from modules.frame.frame import *
from modules.visual_map.map import *
from utils.rotation.dcm import *
import matplotlib.pyplot as plt
from utils.ips.parse import *
from utils.conversion.coordinate  import *
from plyfile import PlyData, PlyElement


def write_ply(points, output_file,text=True):
    points=[
        (0,0,0),
        (0,100,0),
        (100,100,0),
        (100,0,0),
        (50,50,75)
    ]

    face=np.array([
        ((0,1,2),255,0,0),
        ((0,2,3),255,0,0),
        ((0, 1, 4),0,255,0),
        ((1,2,4),0,0,255),
        ((2,3,4),255,255,0),
        ((0,3,4),0,0,0)],
        dtype=[('vertex_index','f4',(3,)),
               ('red','u1'),('green','u1'),
               ('blue','u1')]
    )
    print(face)
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    face = PlyElement.describe(face, 'face')
    PlyData([el,face], text=text).write(output_file)

root_path = "/home/xuzhuo/Documents/code/python/smartpnt_tools/log"
traj_file = f"{root_path}/frame.txt"
points_file = f"{root_path}/points.txt"
ply_file = f"{root_path}/map.ply"

mps = np.loadtxt(points_file)[:, 1:]
# parser = ParseIPSSol()
sol = np.loadtxt(traj_file)
# x, y, z = sol["X-ECEF"], sol["Y-ECEF"], sol["Z-ECEF"]
test = sol[:, 1:]

print(XYZ2NEU(test[0, :], test[-1, :]))
coor_enu = np.array([XYZ2NEU(test[0, :], test[i, :]) for i in range(1, test.shape[0])])
mps_enu = [XYZ2NEU(test[0, :], mps[i, :3]) for i in range(1, mps.shape[0])]

mps_ply = [(mps_enu[i][0], mps_enu[i][1], mps_enu[i][2], int(0), int(255), int(0)) for i in range(1, len(mps_enu))]
coor_ply = [(coor_enu[i][0], coor_enu[i][1], coor_enu[i][2], int(255), int(0), int(0)) for i in range(1, len(coor_enu))]
coor_ply = coor_ply + mps_ply
# total = np.array([mps_ply, coor_ply])
total_ply = np.array(coor_ply, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('blue','u1'),('green','u1'), ('red','u1')])
# coor_ply = np.array(coor_ply, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('blue','u1'),('green','u1'), ('red','u1')])

print(coor_ply[0])

total_vertex = PlyElement.describe(total_ply, 'vertex')
# vertex_traj = PlyElement.describe(coor_ply, 'vertex')
PlyData([total_vertex], text=True).write(ply_file)
