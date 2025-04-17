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
from utils.ips import parse

path_sol = "/home/xuzhuo/Documents/data/gici/new/5.2/IPSProj/R_RTK_LCI/gnss_rover.flf"

color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
        2: [(255 / 255), (102 / 255), (102 / 255)],  # red
        1: [(255 / 255), (146 / 255), (0 / 255)],  # blue
        0: [(0 / 255), (141 / 255), (0 / 255)],
        4: [(20 / 255), (169 / 255), (89 / 255)],
        5: [(70 / 255), (114 / 255), (196 / 255)],
        6: [(150 / 255), (150 / 255), (150 / 255)],
        7: [(200 / 255), (200 / 255), (200 / 255)]}  # green

font={
    #   'family':'Cambria',
      'size': 14, # corresponding to 10 pt
      'weight': 'bold'
}

parser = parse.ParseIPSSol()
header, sol = parser.parseSINS(path_sol)
ba = [sol["AccBiasX"], sol["AccBiasY"], sol["AccBiasZ"]]
label_ba = ["ba-x", "ba-y", "ba-z"]
bg = [sol["GyrBiasX"], sol["GyrBiasY"], sol["GyrBiasZ"]]
label_bg = ["bg-x", "bg-y", "bg-z"]
sd_ba = [sol["SD-AccBiasX"], sol["SD-AccBiasY"], sol["SD-AccBiasZ"]]
sd_bg = [sol["SD-GyrBiasX"], sol["SD-GyrBiasY"], sol["SD-GyrBiasZ"]]

# plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fig, axs = plt.subplots(2, 3, figsize=(10, 5))

logger.info(ba[0].shape[0])
for i, ax in enumerate(axs[0]):
    if i == 0:
        ax.set_ylabel("mg", labelpad=3, fontsize = 13, fontdict=font) 
    ax.grid(linestyle='-', color='k', alpha=0.055)
    ax.plot(sol["GPSTime"], ba[i], ls='-', color=color[0], linewidth=2.5, label=label_ba[i])
    ax.plot(sol["GPSTime"], sd_ba[i], ls='--', color=color[6], linewidth=1.5)
    ax.plot(sol["GPSTime"], -sd_ba[i], ls='--', color=color[6], linewidth=1.5)
    # ax.axvline(179740, linestyle='--', color=color[1])
    ax.legend()
    
for i, ax in enumerate(axs[1]):
    if i == 0:
        ax.set_ylabel("deg/hr", labelpad=3, fontsize = 13, fontdict=font)
    if i == 1:
        ax.set_xlabel("time [s]", labelpad=3, fontsize = 13, fontdict=font)

    ax.grid(linestyle='-', color='k', alpha=0.055) 
    ax.plot(sol["GPSTime"], bg[i], ls='-', color=color[0], linewidth=2.5, label=label_bg[i])
    ax.plot(sol["GPSTime"], sd_bg[i], ls='--', color=color[6], linewidth=1.5)
    ax.plot(sol["GPSTime"], -sd_bg[i], ls='--', color=color[6], linewidth=1.5)
    # ax.axvline(179740, linestyle='--', color=color[1])
    ax.legend()
plt.show()

