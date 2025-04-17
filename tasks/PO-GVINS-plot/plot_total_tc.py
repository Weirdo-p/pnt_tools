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

root = "/home/xuzhuo/Documents/data/01-mini/20240717/proj/R_RTK_TCI"
com_file = os.path.join(root, "ROVE_03.ftf.com")
com_data = np.loadtxt(com_file)

com_data = com_data[(com_data[:, 0] > 286763) & (com_data[:, 0] < 288750), :]
com_data[:, 0] = com_data[:, 0] - com_data[0, 0]
logger.info(com_data)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.3))
direc = ['R', 'F', 'U']

plt.grid(linestyle='-', color='k', alpha=0.14)

for i in range(3):

    direc_ = direc[i]
    plt.plot(com_data[:, 0], (com_data[:, i + 1]) , ls="-", color=color[i], label=direc[i], linewidth=2.5)#, marker=marker[i], markersize=4)

    # print(RMS(np.zeros(direc_.shape), direc_), " m")
plt.ylabel("Error (m)", labelpad=3, fontsize = 13, fontdict=font)
plt.xticks(size = 12)
# legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
plt.margins(x=0, y=0)
# if attribute["xlim"][0] != attribute["xlim"][1]:
#     plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
# if attribute["ylim"][0] != attribute["ylim"][1]: 
#     plt.ylim(attribute["ylim"][0], attribute["ylim"][1])
plt.xlim(0, 2000)
plt.ylim(-3, 3)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
plt.subplots_adjust(left=0.13, right=0.94, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
plt.xlabel("time (s)", fontdict=font)
# plt.savefig("/home/xuzhuo/Documents/data/01-mini/20240717/proj/R_RTK_TCI/TC.svg", transparent=True)
plt.show()

#%% 
plt.close()
# color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
#         2: [(227 / 255), (98 / 255), (93 / 255)],  # red
#         1: [(240 / 255), (194 / 255), (132 / 255)],  # blue
#         0: [(196 / 255), (214 / 255), (159 / 255)],
#         4: [(20 / 255), (169 / 255), (89 / 255)],
#         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green

parser = ParseIPSSol()
head, sol = parser.parseSINS("/home/xuzhuo/Documents/data/01-mini/20240717/proj/R_RTK_TCI/ROVE_03.ftf")
used_data = (sol['GPSTime'] > 286763) & (sol['GPSTime'] < 288750)
testA_time = sol["GPSTime"][used_data]
testA_pdop = np.sqrt(np.linalg.norm(sol["HDOP"][used_data].reshape(-1, 1), axis=1, ord=2) + np.linalg.norm(sol["VDOP"][used_data].reshape(-1, 1), axis=1, ord=2))
testA_NSAT = sol["NSAT"][used_data]
logger.info(testA_pdop)
logger.info(testA_NSAT)
testA_time = testA_time - testA_time[0]
plotPdop(testA_time, testA_pdop, testA_NSAT, "/home/xuzhuo/Documents/data/01-mini/20240717/proj/total_nsat.svg")


# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.3))
# direc = ['R', 'F', 'U']
# for i in range(3):

#     plt.grid(linestyle='-', color='k', alpha=0.2)
#     direc_ = direc[i]
#     plt.plot(com_data[:, 0], (com_data[:, i + 1]) , ls="-", color=color[i], label=direc[i], linewidth=2.5)#, marker=marker[i], markersize=4)

#     # print(RMS(np.zeros(direc_.shape), direc_), " m")
# plt.ylabel("Error (m)", labelpad=3, fontsize = 13, fontdict=font)
# plt.xticks(size = 12)
# # legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
# plt.margins(x=0, y=0)
# # if attribute["xlim"][0] != attribute["xlim"][1]:
# #     plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
# # if attribute["ylim"][0] != attribute["ylim"][1]: 
# #     plt.ylim(attribute["ylim"][0], attribute["ylim"][1])
# plt.xlim(0, 2000)
# plt.ylim(-3, 3)
# ax.spines['bottom'].set_linewidth(1)
# ax.spines['left'].set_linewidth(1)
# ax.spines['right'].set_linewidth(1)
# ax.spines['top'].set_linewidth(1)
# plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
# plt.xlabel("time (s)", fontdict=font)
# plt.savefig("/home/xuzhuo/Documents/data/01-mini/20240717/proj/R_RTK_TCI/TC.svg", transparent=True)
# plt.show()
