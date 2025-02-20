import numpy as np
import matplotlib.pyplot as plt

# 设置参数
num_samples = 1000  # 样本数量
fs = 1000           # 采样频率(Hz)
t = np.arange(num_samples) / fs  # 时间轴

# 生成高斯白噪声
gaussian_noise = np.random.normal(0, 1, num_samples)

# 生成布朗噪声（随机游走）
brown_noise = np.cumsum(gaussian_noise)
brown_noise = brown_noise / np.std(brown_noise)  # 标准化

# 创建图形
plt.figure(figsize=(4.0393701, 2.))

# 绘制两个噪声在同一子图中
plt.plot(t, gaussian_noise, 'b-', linewidth=0.5, alpha=0.7, label='Gaussian White Noise')
plt.plot(t, brown_noise, 'r-', linewidth=0.8, alpha=0.8, label='Brownian Noise')
ax = plt.gca()
# 去除所有边框和刻度
ax.set_xticks([])
ax.set_yticks([])
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_visible(False)
# 添加图例和标签
# plt.legend(fontsize=10)
# plt.title('Comparison of Gaussian White Noise and Brownian Noise', fontsize=12)
# plt.xlabel('Time (s)', fontsize=10)
# plt.ylabel('Amplitude', fontsize=10)
plt.xticks([])
plt.yticks([])
plt.grid(True, alpha=0.3)

# 美化图形
plt.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig("./noise.svg", transparent=True)

plt.show()

