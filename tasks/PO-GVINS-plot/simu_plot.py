import sys
import os, glob
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.logger.logger import logger
from utils.plot_style.style import *
import matplotlib.pyplot as plt


root_path = "/home/xuzhuo/Documents/data/simu"
plot_name = "use_all_point"
file_path = os.path.join(root_path, plot_name)

result_files = os.listdir(file_path)

msckf = [file for file in result_files if "msckf" in file]
po = [file for file in result_files if "poseonly" in file]

msckf_direc:dict[str, dict[str, list]] = {}
for file in msckf:
    direc = file.split("_")[2]
    assert direc in ['x', 'y', 'z']
    if msckf_direc.get(direc) is None:
        msckf_direc[direc] = {}
    task = "att" if "att" in file else "pos"
    if msckf_direc[direc].get(task) is None:
        msckf_direc[direc][task] = []
    msckf_direc[direc][task].append(file)
    msckf_direc[direc][task] = sorted(msckf_direc[direc][task], key=lambda x: float(x.split("_")[3]))

po_direc:dict[str, dict[str, list]] = {}
for file in po:
    direc = file.split("_")[2]
    assert direc in ['x', 'y', 'z']
    if po_direc.get(direc) is None:
        po_direc[direc] = {}
    task = "att" if "att" in file else "pos"
    if po_direc[direc].get(task) is None:
        po_direc[direc][task] = []
    po_direc[direc][task].append(file)
    po_direc[direc][task] = sorted(po_direc[direc][task], key=lambda x: float(x.split("_")[3]))

print(msckf_direc)
print(po_direc)

direcs, tasks = ['x', 'y', 'z'], ['att', 'pos']
for direc in direcs:
    for task in tasks:
        msckf_files = msckf_direc[direc][task]
        po_files = po_direc[direc][task]
        # logger.info(msckf_files)
        # logger.info(po_files)
        for msckf_file, po_file in zip(msckf_files, po_files):
            msckf_file_path = os.path.join(file_path, msckf_file)
            po_file_path = os.path.join(file_path, po_file)
            logger.info(msckf_file_path)
            # logger.info(po_file_path)
            msckf_data = np.loadtxt(msckf_file_path)
            po_data = np.loadtxt(po_file_path)
            # logger.info(msckf_data.shape)
            # logger.info(po_data.shape)
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.4645669))
            msckf_pos_error = np.linalg.norm(msckf_data[:, 25:28], axis=1)
            po_pos_error = np.linalg.norm(po_data[:, 25:28], axis=1)
            logger.info([msckf_pos_error, po_pos_error])
            # ax.plot(msckf_data[:, 0], msckf_data[:, 1], label="msckf", color=colors[0])

#             plt.figure()
#             plt.plot(msckf_data[:, 0], msckf_data[:, 1], label="msckf")
#             plt.plot(po_data[:, 0], po_data[:, 1], label="poseonly")
#             plt.legend()
#             plt.show()
# # msckf = sorted(msckf, key=lambda x: float(x.split("_")[3]))
# # po = sorted(po, key=lambda x: float(x.split("_")[3]))
# # logger.info(msckf)
# # logger.info(po)