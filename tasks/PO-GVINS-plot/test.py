import numpy as np
import matplotlib.pyplot as plt

# 示例数据：3个类别，每个类别是一个1000x4的numpy数组
np.random.seed(0)
data_list = [np.hstack((np.random.uniform(0, 500, (1000, 1)), np.random.rand(1000, 3))) for _ in range(3)]
# print(data_list)
# 定义距离分组区间
bin_size = 100
bins = np.arange(0, 501, bin_size)
print(bins)
# 存储每个区间的数据
all_grouped_data = []

for data in data_list:
    grouped_data = []
    for i in range(len(bins) - 1):
        # 获取当前区间的掩码
        mask = (data[:, 0] >= bins[i]) & (data[:, 0] < bins[i + 1])
        # 获取当前区间的数据（去掉距离列）
        group = data[mask, 1:].flatten()  # 展平数据方便箱状图绘制
        if group.size == 0:
            group = np.array([np.nan])  # 如果当前区间没有数据，用NaN填充
        grouped_data.append(group)
    all_grouped_data.append(grouped_data)

# 为每个类别绘制箱状图
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'black']  # 不同类别的颜色
width = 0.2  # 箱状图的宽度

for i, grouped_data in enumerate(all_grouped_data):
    # 计算每个类别在每个区间的水平位置
    print(np.arange(len(bins) - 1) + i * width - width)
    positions = np.arange(len(bins) - 1) + i * width - width
    plt.boxplot(grouped_data, positions=positions, widths=width, patch_artist=True, 
                boxprops=dict(facecolor=colors[i], color=colors[i]),
                medianprops=dict(color='black'),
                showmeans=True, meanline=True, 
                flierprops=dict(markerfacecolor=colors[i], marker='o', markersize=5, linestyle='none'))

# 设置图例和轴标签
plt.xticks(ticks=np.arange(len(bins) - 1), labels=[f'{bins[i]}-{bins[i + 1]}m' for i in range(len(bins) - 1)])
plt.xlabel('Distance [m]')
plt.ylabel('Data Values')
plt.title('Boxplot of Data by Distance Range')
plt.legend([f'Category {i+1}' for i in range(len(data_list))], loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
