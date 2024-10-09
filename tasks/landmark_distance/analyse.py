# %% import libraries
import numpy as np
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from modules.camera.camera import *
from modules.frame.frame import *
from modules.visual_map.map import *
from utils.rotation.dcm import *
import matplotlib.pyplot as plt
font={
    #   'family':'Cambria',
      'size': 14, # corresponding to 10 pt
      'weight': 'bold'
}

#%% set basic parameter
euroc_depth_file = "/home/xuzhuo/Documents/code/C++/OpenSource/open_vins/log/depth.txt"
euroc_data = np.loadtxt(euroc_depth_file)
euroc_data = sorted(euroc_data)
logger.info(len(euroc_data))
euroc_prob = np.array(list(range(1, len(euroc_data)))) / len(euroc_data)
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
# fig, ax = plt.subplots(1, 1, figsize=(3.0393701, 2.4645669))
fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.4645669))
plt.hist(euroc_data, edgecolor='w')
plt.margins(x=0, y=0)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.ylabel("Probability", labelpad=3, fontsize = 13, fontdict=font)
plt.xlabel("Depth (m)", labelpad=3, fontsize = 13, fontdict=font)
plt.ylim(0, 600)
plt.xlim(0, 50)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
plt.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
plt.savefig("euroc.svg", transparent=True)
# plt.show()
plt.close()
logger.info(euroc_prob)

#%%
mini_depth_file = "/home/xuzhuo/Documents/code/C++/IPS-CMake-BuildProject-Linux/log/depth_mini.txt"
mini_data = np.loadtxt(mini_depth_file)
mini_data = sorted(mini_data)
mini_data = [x for x in mini_data if x > 0]
mini_data_1 = mini_data[0:6000]
logger.info(len(mini_data_1))
fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.4645669))
plt.hist(mini_data_1, edgecolor='w', bins=20)
plt.margins(x=0, y=0)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.ylabel("Probability", labelpad=3, fontsize = 13, fontdict=font)
plt.xlabel("Depth (m)", labelpad=3, fontsize = 13, fontdict=font)
plt.ylim(0, 1000)
plt.xlim(0, 50)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
plt.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
# plt.show()
plt.savefig("mini.svg", transparent=True)

f = 110 * 1E-4
pixel_size = 3.76 * 1E-6
fx = f / pixel_size
fy = fx
cx = cy = 5000
width, height = 20000, 13000
logger.info(f"camera parameters fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
camera = Camera(fx, fy, cx, cy, [width, height], type=CameraType.MONO)

#%% set frame parameter
bpos = np.zeros((3, 1))
bpos[0] = 1
frame_a, frame_b = Frame(0), Frame(1, bpos)

plane_pos = np.array([[0.75], [0], [30]])
plane = MapPoint(plane_pos, 0)

plane_pos_a = frame_a.project_to_image(plane, camera)
plane_pos_cam_a = frame_a.project_to_camera(plane)
logger.info(f"plane pos in frame_a is {plane_pos_a.flatten()}")
plane_feat_a = Feature(plane_pos_a, 0, 0)
plane_feat_a.m_PosInCamera = plane_pos_cam_a
plane_feat_a.m_frame = frame_a

plane_pos_b = frame_b.project_to_image(plane, camera)
plane_pos_cam_b = frame_b.project_to_camera(plane)
logger.info(f"plane pos in frame_b is {plane_pos_b.flatten()}")
plane_feat_b = Feature(plane_pos_b, 0, 0)
plane_feat_b.m_frame = frame_b
plane_feat_b.m_PosInCamera = plane_pos_cam_b

plane.m_obs.append(plane_feat_a)
plane.m_obs.append(plane_feat_b)


# missile_pos = np.array([[0.5], [0], [13]]) * 1E3
# missile = MapPoint(missile_pos, 0)

# missile_pos_a = frame_a.project_to_image(missile, camera)
# missile_pos_cam_a = frame_a.project_to_camera(missile)
# logger.info(f"missile pos in frame_a is {missile_pos_a.flatten()}")
# missile_feat_a = Feature(missile_pos_a, 0, 0)
# missile_feat_a.m_PosInCamera = missile_pos_cam_a
# missile_feat_a.m_frame = frame_a

# missile_pos_b = frame_b.project_to_image(missile, camera)
# missile_pos_cam_b = frame_b.project_to_camera(missile)
# logger.info(f"missile pos in frame_a is {missile_pos_b.flatten()}")
# missile_feat_b = Feature(missile_pos_b, 0, 0)
# missile_feat_b.m_PosInCamera = missile_pos_cam_b
# missile_feat_b.m_frame = frame_b

# missile.m_obs.append(missile_feat_a)
# missile.m_obs.append(missile_feat_b)
#%% triangulation without error simulated
plane_pos_t = plane.triangulate()
logger.warning("triangulation error {}, triangulated pos: {}, theoretical pos: {}".format((plane_pos_t - plane.m_pos).flatten(), plane_pos_t.flatten(), plane.m_pos.flatten()))

# missile_pos_t = missile.triangulate()
# logger.warning("triangulation error {}, triangulated pos: {}, theoretical pos: {}".format((missile_pos_t - missile.m_pos).flatten(), missile_pos_t.flatten(), missile.m_pos.flatten()))

# dist_t = np.linalg.norm(plane_pos_t - missile_pos_t)
# dist_o = np.linalg.norm(plane.m_pos - missile.m_pos)
# error = np.abs(dist_t - dist_o)
# logger.warning(f"distance between plane and missile is {dist_t}, theorectically: {dist_o}, error is {error}")

#%% disturbance analysis
PIXEL_EXTRACT_ERROR = np.array([0.5, 0.5, 0]).reshape(3, 1)
ATTITUDE_ERROR = np.array([0, 0, 0.05]) * np.pi / 180.0
POS_ERROR = np.array([0.01, 0, 0]).reshape(3, 1)


#%% 4. all errors added
logger.info(f"adding {PIXEL_EXTRACT_ERROR} extract errors to plane and missile, {POS_ERROR.flatten()}m translation errors to frame b and {ATTITUDE_ERROR.flatten()} rad attitude errors to frame b")
for i in range(len(plane.m_obs)):
    plane.m_obs[i].m_pos = plane.m_obs[i].m_pos + PIXEL_EXTRACT_ERROR
    plane.m_obs[i].m_PosInCamera = camera.lift(plane.m_obs[i].m_pos)

frame_b.m_pos = frame_b.m_pos + POS_ERROR
frame_b.m_rota = Angle2RotMatrix(ATTITUDE_ERROR)

plane_pos_t = plane.triangulate()
logger.warning("triangulation error {}, triangulated pos: {}, theoretical pos: {}".format((plane_pos_t - plane.m_pos).flatten(), plane_pos_t.flatten(), plane.m_pos.flatten()))
