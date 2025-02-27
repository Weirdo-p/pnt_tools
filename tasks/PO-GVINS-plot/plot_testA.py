import sys
import os, glob
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from utils.plot_style.style import *
import matplotlib.pyplot as plt
from utils.ips.parse import *

color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
        2: [(255 / 255), (102 / 255), (102 / 255)],  # red
        1: [(255 / 255), (146 / 255), (0 / 255)],  # blue
        0: [(0 / 255), (141 / 255), (0 / 255)],
        4: [(20 / 255), (169 / 255), (89 / 255)],
        5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

# color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
#         2: [(212 / 255), (86 / 255), (46 / 255)],
#         1: [(219 / 255), (180 / 255), (40 / 255)],  # blue
#         0: [(132 / 255), (186 / 255), (66 / 255)],
#         4: [(20 / 255), (169 / 255), (89 / 255)],
#         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

root = "/home/xuzhuo/Documents/data/01-mini/20240717/proj"
GVIO_PO_file = os.path.join(root, "287000-287090_GVIO_PO/ROVE_04.flf.imu_VIO.cmp")
GVIO_PO = np.loadtxt(GVIO_PO_file)
MSCKF_GVIO_file = os.path.join(root, "287000-287090_GVIO_MSCKF/ROVE_04.flf.imu_VIO.cmp")
MSCKF_GVIO = np.loadtxt(MSCKF_GVIO_file)
# GVIO_PO = GVIO_PO[::2, :]
GI_TC_file = os.path.join(root, "287000-287090_LI_TC/ROVE_04.flf.imu_VIO.cmp")
GI_TC = np.loadtxt(GI_TC_file)
# GI_TC = GI_TC[::2, :]
PO_file = os.path.join(root, "287000-287090_PO/ROVE_04.flf.imu_VIO.cmp")
PO = np.loadtxt(PO_file)
MSCKF_file = os.path.join(root, "287000-287090_MSCKF/ROVE_04.flf.imu_VIO.cmp")
MSCKF = np.loadtxt(MSCKF_file)
MSCKF[:, 0] -= MSCKF[0, 0]
PO[:, 0] -= PO[0, 0]
GNSS_file = os.path.join(root, "R_RTK/ROVE_04.ffp.com")
# MSCKF_GVIO_file = os.path.join(root, "R_RTK_TCI/ROVE_04.flf.imu_VIO.cmp")
# MSCKF_GVIO = np.loadtxt(MSCKF_GVIO_file)

# GNSS = np.loadtxt(GNSS_file)
# GNSS = GNSS[(GNSS[:, 0] > 287000) & (GNSS[:, 0] < 287090), :]
all_data = [GI_TC, MSCKF_GVIO, GVIO_PO]
markers = ["o", "o", "o"]
labels = ["GI", "MSCKF-GVINS", "PO-GVINS"]

header, sol = ParseIPSSol().parseSINS("/home/xuzhuo/Documents/data/01-mini/20240717/proj/R_RTK_TCI/ROVE_04.ftf")
used_data = (sol['GPSTime'] > 287000) & (sol['GPSTime'] < 287090)
testA_time = sol["GPSTime"][used_data]
testA_time = testA_time - testA_time[0]
testA_pdop = np.sqrt(np.linalg.norm(sol["HDOP"][used_data].reshape(-1, 1), axis=1, ord=2) + np.linalg.norm(sol["VDOP"][used_data].reshape(-1, 1), axis=1, ord=2))
testA_NSAT = sol["NSAT"][used_data]

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fig, ax = plt.subplots(4, 1)
plt.subplots_adjust(left=0.1,bottom=0.05,top=0.95, right=0.95)#调整子图间距
direc = ['Right', 'Front', 'Up']
legends = []

color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
        2: [(255 / 255), (102 / 255), (102 / 255)],  # red
        1: [(114 / 255), (176 / 255), (99 / 255)],  # blue
        0: [(226 / 255), (145 / 255), (53 / 255)],
        4: [(20 / 255), (169 / 255), (89 / 255)],
        5: [(70 / 255), (114 / 255), (196 / 255)]}  # green
color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
        2: [(255 / 255), (102 / 255), (102 / 255)],  # red
        0: [(250 / 255), (164 / 255), (25 / 255)],  # blue
        1: [(77 / 255), (183 / 255), (72 / 255)],  # blue
        4: [(20 / 255), (169 / 255), (89 / 255)],
        5: [(70 / 255), (114 / 255), (196 / 255)]}  # green
color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
        0: [(252 / 255), (102 / 255), (102 / 255)],  # red
        1: [(250 / 255), (164 / 255), (25 / 255)],  # blue
        2: [(77 / 255), (183 / 255), (72 / 255)],  # blue
        4: [(20 / 255), (169 / 255), (89 / 255)],
        5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

for i_th, data in enumerate(all_data):
    for i in range(3):
        ax[i].grid(linestyle='-', color='k', alpha=0.14)
        test = ax[i].plot(data[:, 0] - data[0, 0], (data[:, i + 1]) , ls="-", label=labels[i_th], linewidth=1, marker=markers[i_th], markersize=5, color=color[i_th], mec=[0,0,0], markeredgewidth=0.7)#, marker=marker[i], markersize=4)
        # ax[i].legend()
        ax[i].set_xlim(0, 90)
        ax[i].set_ylim(-4, 4)
        ax[i].spines['bottom'].set_linewidth(1.5)
        ax[i].spines['left'].set_linewidth(1.5)
        ax[i].spines['right'].set_linewidth(1.5)
        ax[i].spines['top'].set_linewidth(1.5)

        # ax[i].set_xticks()
        ax[i].tick_params(axis='both', which='major', labelsize=12)
        ax[i].set_yticks([-4, -2, 0, 2, 4])
        if i == 2:
            ax[i].set_ylim(-2, 2)
            ax[i].set_yticks([-2, -1, 0, 1, 2])

        ax[i].axes.xaxis.set_ticklabels([])
        ax[i].spines['bottom'].set_linewidth(1)
        ax[i].spines['left'].set_linewidth(1)
        ax[i].spines['right'].set_linewidth(1)
        ax[i].spines['right'].set_linewidth(1)



color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
        2: [(255 / 255), (102 / 255), (102 / 255)],  # red
        1: [(255 / 255), (146 / 255), (0 / 255)],  # blue
        0: [(0 / 255), (141 / 255), (0 / 255)],
        4: [(20 / 255), (169 / 255), (89 / 255)],
        5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

ax[3].tick_params(axis='both', which='major', labelsize=12)
ax[3].scatter(testA_time, testA_pdop, linewidths=1.5, color=color[1],s=10, marker="^")
ax[3].spines['left'].set_color([(238 / 255), (159 / 255), (0 / 255)])

for i in range(len(ax[3].get_yticklabels())):
    ax[3].get_yticklabels()[i].set_color([(238 / 255), (159 / 255), (0 / 255)])
ax[3].tick_params(width=1.5, axis='y', which='minor',colors=[(238 / 255), (159 / 255), (0 / 255)])
ax[3].tick_params(width=1.5, axis='y', color=[(238 / 255), (159 / 255), (0 / 255)])
ax[3].tick_params(width=1.5, axis='x')
ax[3].set_ylim(0, 20)
ax[3].grid(linestyle='-', color='k', alpha=0.14)

ax = ax[3].twinx()
ax.scatter(testA_time, testA_NSAT, linewidths=1.5, color=color[0],s=10, marker="s")
ax.set_ylim(20, 40)
ax.tick_params(axis='both', which='major', labelsize=12)

for i in range(len(ax.get_yticklabels())):
    ax.get_yticklabels()[i].set_color(color[0])
ax.tick_params(width=1.5, axis='y', color=color[0])
ax.spines['right'].set_color(color[0])
ax.spines['left'].set_color([(238 / 255), (159 / 255), (0 / 255)])
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.set_xlim(0, 90)
plt.savefig("/home/xuzhuo/Documents/data/01-mini/20240717/proj/287000-287090_GVIO_GI.svg", transparent=True)
plt.show()

MSCKF_cdf = np.linalg.norm(MSCKF[:, 1:4], axis=1, ord=2)
PO_cdf = np.linalg.norm(PO[:, 1:4], axis=1, ord=2)
GI_TC_cdf = np.linalg.norm(GI_TC[:, 1:4], axis=1, ord=2)
GVIO_PO_cdf = np.linalg.norm(GVIO_PO[:, 1:4], axis=1, ord=2)
GVIO_M_CDF = np.linalg.norm(MSCKF_GVIO[:, 1:4], axis=1, ord=2)
all_data = [GI_TC_cdf, GVIO_PO_cdf, GVIO_M_CDF]
# all_data = [MSCKF_cdf, PO_cdf, GI_TC_cdf, GVIO_PO_cdf, GVIO_M_CDF]
# labels = ["MSCKF", "PO-VINS", "GI", "PO-GVINS", "M-GVINS"]
labels = ["GI", "PO-GVINS", "M-GVINS"]

# gt_file = os.path.join(root, "I300.imr.ref")
color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
        0: [(252 / 255), (102 / 255), (102 / 255)],  # red
        2: [(250 / 255), (164 / 255), (25 / 255)],  # blue
        1: [(77 / 255), (183 / 255), (72 / 255)],  # blue
        4: [(20 / 255), (169 / 255), (89 / 255)],
        5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

CDF_plot = dict(zip(labels, all_data))
# plotCDF(CDF_plot, "/home/xuzhuo/Documents/data/01-mini/20240717/proj/testA_GNSSrelated_cdf.svg", color=color)
logger.info(CDF_plot)
all_data = [MSCKF_cdf, PO_cdf]
labels = ["MSCKF", "PO-VINS"]
CDF_plot = dict(zip(labels, all_data))
color = {3: [(0 / 255), (167 / 255), (251 / 255)],  # black
        2: [(252 / 255), (169 / 255), (0 / 255)],  # red
        1: [(100 / 255), (13 / 255), (171 / 255)],  # blue
        0: [(245 / 255), (82 / 255), (0 / 255)],
        4: [(20 / 255), (169 / 255), (89 / 255)],
        5: [(70 / 255), (114 / 255), (196 / 255)]}  # green
plotCDF(CDF_plot, "/home/xuzhuo/Documents/data/01-mini/20240717/proj/testA_vis_only_cdf.svg", color=color)

def getIPSPose(file, time_range):
    parser = ParseIPSSol()
    head, sol = parser.parseSINS(file)
    used_data = (sol['GPSTime'] >= time_range[0]) & (sol['GPSTime'] < time_range[1])
    pose = np.zeros((used_data.shape[0], 7))
    pose[:, 0] = sol['GPSTime'][used_data]
    pose[:, 1] = sol['X-ECEF'][used_data]
    pose[:, 2] = sol['Y-ECEF'][used_data]
    pose[:, 3] = sol['Z-ECEF'][used_data]
    pose[:, 4] = sol['Yaw'][used_data]
    pose[:, 5] = sol['Pitch'][used_data]
    pose[:, 6] = sol['Roll'][used_data]
    pose_copy = np.zeros(pose.shape)
    j, count = 0, 0
    for i in range(pose.shape[0]):
        if i == 0:
            pose_copy[j, :] = pose[i, :]
            count += 1
            j += 1
            continue
        if pose[i, 0] - pose_copy[j - 1, 0] > 0:
            pose_copy[j, :] = pose[i, :]
            j += 1
            count += 1
    pose_copy = pose_copy[:count, :]
    return pose_copy

import utils.rotation.dcm as dcm
import utils.conversion.coordinate as coor
def getRelativeError(IpsSol, IEsol):
    assert IpsSol.shape[0] == IEsol.shape[0]
    relative_error = np.zeros((IpsSol.shape[0], 4))
    i = 1
    while i < IpsSol.shape[0]:
        assert IpsSol[i, 0] == IEsol[i, 0]
        relative_error[i, 0] = IpsSol[i, 0]
        relative_error[i, -1] = IE[i, -1]
        Ips_pos_interval = IpsSol[i, 1: 4] - IpsSol[i - 1, 1: 4]
        IE_pos_interval = IEsol[i, 1: 4] - IEsol[i - 1, 1: 4]
        relative_error[i, 1] = np.linalg.norm(Ips_pos_interval - IE_pos_interval, ord=2)

        ips_rotation = dcm.Angle2RotMatrix(dcm.Azimuth2Attitude(IpsSol[i, 4: 7] * np.pi / 180.0))
        ips_rotation_prev = dcm.Angle2RotMatrix(dcm.Azimuth2Attitude(IpsSol[i - 1, 4: 7] * np.pi / 180.0))
        ie_rotation = dcm.Angle2RotMatrix(dcm.Azimuth2Attitude(IEsol[i, 4: 7] * np.pi / 180.0))
        ie_rotation_prev = dcm.Angle2RotMatrix(dcm.Azimuth2Attitude(IEsol[i - 1, 4: 7] * np.pi / 180.0))
        relative_error[i, 2] = np.linalg.norm(dcm.RotMatrix2RotAngle(ips_rotation_prev.transpose() @ ips_rotation @ (ie_rotation_prev.transpose() @ ie_rotation).transpose()) * 180 / np.pi)
        # dcm.
        logger.info(coor.Rbe2Attitude(
            coor.Attitude2Rbe(
                dcm.Azimuth2Attitude(IEsol[i, 4: 7] * np.pi / 180.0),
                    IEsol[i, 1: 4]), 
                coor.XYZ2BLH(IEsol[i, 1: 4])).flatten() - dcm.Azimuth2Attitude(IEsol[i, 4: 7] * np.pi / 180.0).flatten())
        # logger.info(f"{relative_error[i, 1]}, {relative_error[i, 2]}")
        i += 1
    plt.figure()
    plt.plot(relative_error[:, -1], relative_error[:, 1], label = "pos")
    plt.plot(relative_error[:, -1], relative_error[:, 2], label = "att")
    plt.legend()
    plt.ylim(0, 3)
    plt.show()
    pass

root = "/home/xuzhuo/Documents/data/01-mini/20240717/proj"
GVIO_PO_file = os.path.join(root, "287000-287090_GVIO_PO/ROVE_04.flf")
GVIO_PO = getIPSPose(GVIO_PO_file, [287000, 287090])
GI_TC_file = os.path.join(root, "287000-287090_LI_TC/ROVE_04.flf")
GI_TC = getIPSPose(GI_TC_file, [287000, 287090])
PO_file = os.path.join(root, "287000-287090_PO/ROVE_04.flf")
PO = getIPSPose(PO_file, [287000, 287090])
MSCKF_file = os.path.join(root, "287000-287090_MSCKF_one_iter/ROVE_04.flf")
MSCKF = getIPSPose(MSCKF_file, [287000, 287090])
MSCKF_GVINS_file = os.path.join(root, "287000-287090_GVIO_MSCKF/ROVE_04.flf")
MSCKF_GVIO = getIPSPose(MSCKF_GVINS_file, [287000, 287090])



IE_File = os.path.join(root, "I300_GroundTruth-10HZ.txt")
IE_all = np.loadtxt(IE_File, dtype=str, skiprows=28)
IE = np.zeros((IE_all.shape[0], 8))
IE[:, 0] = IE_all[:, 1].astype(np.float64)
IE[:, 1: 4] = IE_all[:, 9: 12].astype(np.float64)
IE[:, 4: 7] = IE_all[:, 21: 24].astype(np.float64)
IE[:, 7] = IE_all[:, -1].astype(np.float64)
IE = IE[(IE[:, 0] >= 287000) & (IE[:, 0] < 287090), :]
IE[:, 7] -= IE[0, 7]
IE[:, 7] *= 1000
# print(IE)
logger.info(IE.shape)
# getRelativeError(MSCKF, IE)

from scipy.spatial.transform import Rotation as R
import scipy
print(scipy.__version__)
def write_tum(data, path):
    i = 0
    open(path, "w")

    with open(path, "a") as f:
        while i < data.shape[0]:
            pos = data[i, 1: 4].flatten()
            rota = coor.Attitude2Rbe(dcm.Azimuth2Attitude(data[i, 4: 7] * np.pi / 180.0),data[i, 1: 4])
            rotation_matrix = R.from_matrix(rota)
            # print(rota)
            quat = rotation_matrix.as_quat() # x, y, z, w
            f.write(f"{data[i, 0]} {pos[0]} {pos[1]} {pos[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n")
            i += 1

write_tum(PO, "/home/xuzhuo/Documents/data/01-mini/20240717/proj/PO.tum")
write_tum(MSCKF, "/home/xuzhuo/Documents/data/01-mini/20240717/proj/MSCKF.tum")
write_tum(IE, "/home/xuzhuo/Documents/data/01-mini/20240717/proj/IE.tum")
write_tum(GVIO_PO, "/home/xuzhuo/Documents/data/01-mini/20240717/proj/GVIO_PO.tum")
write_tum(GI_TC, "/home/xuzhuo/Documents/data/01-mini/20240717/proj/GI_TC.tum")
write_tum(MSCKF_GVIO, "/home/xuzhuo/Documents/data/01-mini/20240717/proj/GVIO_MSCKF.tum")

items = ["MSCKF", "PO"]
dists = [1, 5, 10, 20, 40]
root = "/home/xuzhuo/Documents/data/01-mini/20240717/proj/testA/angle"
results:dict[str: dict[int: np.ndarray]] = {}
for item in items:
    results[item] = {}
    for dist in dists:
        folder_path = os.path.join(root, f"{item}_{dist}m")
        errors = np.load(os.path.join(folder_path, "error_array.npy"))
        results[item][dist] = errors

colors = {
    "MSCKF": [(250 / 255), (164 / 255), (25 / 255)],
    "PO": [(77 / 255), (183 / 255), (72 / 255)]
}
categories = list(results.keys())
distances = sorted({dist for category in results.values() for dist in category.keys()})  # 获取所有距离

# 创建图形
plt.figure(figsize=(5.0393701, 3.3))
width = 0.2  # 每个箱状图的宽度

# 遍历类别，添加箱状图
for i, category in enumerate(categories):
    grouped_data = []
    for dist in distances:
        if dist in results[category]:
            grouped_data.append(results[category][dist].flatten())  # 展平数据方便绘制箱状图
        else:
            grouped_data.append([np.nan])  # 如果没有数据，用 NaN 填充

    # 计算箱状图的水平位置
    positions = np.arange(len(distances)) + i * width - width / 2
    plt.boxplot(
        grouped_data,
        positions=positions,
        widths=width,
        patch_artist=True,
        boxprops={'facecolor':colors[category], 'color':"black", 'edgecolor':'black', 'linewidth':1},
        medianprops=dict(color='black'),
        flierprops=dict(markerfacecolor=colors[category], marker='o', markersize=5, linewidth=1)
    )

# 设置图例和轴标签
# plt.ylim(0, 10)
plt.xticks(ticks=np.arange(len(distances)), labels=[f'{dist}' for dist in distances], size = 12)
plt.yticks(size = 12)
# plt.legend([f'{cat}' for cat in categories], loc='upper right', fontsize=8)
plt.grid(axis='y', linestyle='-', alpha=0.17)
plt.tight_layout()
plt.savefig("/home/xuzhuo/Documents/data/01-mini/20240717/proj/testA/angle/testA_angle_rpe.svg", transparent=True)

plt.show()