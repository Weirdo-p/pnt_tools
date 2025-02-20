import numpy as np
import matplotlib.pyplot as plt

# 使用numpy读取CSV文件
data = np.loadtxt('/home/xuzhuo/Documents/nyp/fake.csv', delimiter=',', encoding='utf-8-sig')  # 假设文件是逗号分隔的

# 分离数据和属性
# 第一列是数据，第二列是属性
values = data[:, 0]  # 数据列
attributes = data[:, 1]  # 属性列

# 将数据按属性分组
grouped_data = [values[attributes == attr] for attr in np.unique(attributes)]

# 定义属性到字符串的映射
attribute_mapping = {
    1: 'PDMCI',
    2: 'PDNC',
    3: 'HC'
}

# 获取映射后的标签
labels = [attribute_mapping[attr] for attr in np.unique(attributes)]

# 绘制箱状图
plt.figure(figsize=(3.4645669, 3.4645669))
# 手动设置箱形图的位置，调整间距并使第一组更靠近Y轴
positions = [0.5, 1.1, 1.7]  # 将第一组位置设置为0.8，更靠近Y轴

# 绘制箱状图
boxplot = plt.boxplot(grouped_data, positions=positions, labels=labels, patch_artist=True, widths=0.3)  # widths控制箱形图的宽度

# 设置刻度朝内
plt.tick_params(axis='both', direction='in', labelsize=11, width=1)

# 设置x轴和y轴线的粗细
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)

# 设置箱体颜色
colors = ['lightblue', 'lightgreen', 'lightpink']  # 分别为PDMCI, PDNC, HC设置颜色
for box, color in zip(boxplot['boxes'], colors):
    box.set_facecolor(color)  # 设置箱体颜色
    box.set_linewidth(1.5)  # 设置箱体边框粗细

# 设置均值线的颜色
meanline_color = 'black'  # 均值线颜色
for mean in boxplot['means']:
    mean.set(color=meanline_color, linewidth=0.5)  # 设置均值线颜色和线宽

# 设置中位线的颜色
medianline_color = 'black'  # 中位线颜色
for median in boxplot['medians']:
    median.set(color=medianline_color, linewidth=1.5)  # 设置中位线颜色和线宽

# 设置上下分位线（须线）的粗细
for whisker in boxplot['whiskers']:
    whisker.set_linewidth(1.5)  # 设置须线粗细

# 设置异常值的粗细和样式
for flier, color in zip(boxplot['fliers'], colors):
    flier.set(marker='o', markersize=6, markerfacecolor=color, markeredgecolor='black', markeredgewidth=1.5)  # 设置异常值样式

# 设置刻度字体加粗
plt.xticks(fontweight='bold')
plt.yticks([1.5, 2, 2.5, 3, 3.5],fontweight='bold')

# 隐藏上边和右边的边框线
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 调整X轴范围，使图表布局更紧凑
plt.xlim(0.1, 2)  # 根据positions调整X轴范围
plt.ylim(1.5, 3.5)

# # 设置标题和标签
# plt.xlabel('属性', fontsize=12, fontweight='bold')
# plt.ylabel('数据', fontsize=12, fontweight='bold')

# 显示图表
# plt.show()
plt.savefig('/home/xuzhuo/Documents/nyp/fake_plot.svg', transparent=True)