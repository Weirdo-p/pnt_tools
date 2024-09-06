import rosbag
import glob
import matplotlib.pyplot as plt
import numpy as  np
np.set_printoptions(threshold=np.inf)

prefix = "/home/xuzhuo/Documents/data/01-mini/debug_junce/20220610/20220610/"
bagfiles = glob.glob(prefix + "*bag")

bagfiles = sorted(bagfiles, key=lambda name: float(name[len(prefix)+20: len(name) - 4]))
print(bagfiles)

t_all = []
for bagfile in bagfiles:

    bagdata = rosbag.Bag(bagfile, "r")

    topic = "/cam00/image_raw"

    data = bagdata.read_messages(topic)

    for topic, msg, t in data:
        t_all.append(msg.header.stamp.to_sec())

t_dif = []

for i in range(1, len(t_all)):
    t_dif.append((t_all[i] - t_all[i-1]))

print(t_all[-1] - t_all[0])
print(np.max(t_dif)/1E6)
plt.scatter(range(len(t_dif)), t_dif)
plt.ylim(0.098, 0.102)
plt.show()
np.savetxt("/home/xuzhuo/Documents/code/python/rosbagtool/junce_time_interval_cam.txt", t_dif,fmt="%6.2f")

# test = np.loadtxt("/home/xuzhuo/Documents/code/python/rosbagtool/junce_time_interval.txt")
# plt.figure(1)
# plt.scatter(range(len(test)), test)
# # plt.ylim(4998, 5002)

# plt.show()