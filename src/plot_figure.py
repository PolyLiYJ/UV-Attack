import matplotlib.pyplot as plt
import numpy as np

# 数据准备
categories = ['RCNN', 'YOLOv', 'DETR', 'Retina', 'YOLOv8']
confidences = [0.1, 0.3, 0.5, 0.7, 0.9]

# 创建示例数据，随机生成每个模型在不同置信度下的攻击成功率
np.random.seed(0)
data = np.random.rand(5, 5) * 100  # 将数据缩放到百分比

# 配色方案与提供的图中一致
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 设置字体
plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})

# 绘制直方图
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.15
bar_positions = np.arange(len(categories))

for i, confidence in enumerate(confidences):
    plt.bar(bar_positions + i * bar_width, data[:, i], color=colors[i], width=bar_width, label=f'Conf {confidence}')

# 添加图例和标签
plt.xlabel('Models', fontsize=14)
plt.ylabel('Attack Success Rate (%)', fontsize=14)
plt.title('Attack Success Rate by Model and Confidence Level', fontsize=16)
plt.xticks(bar_positions + bar_width * 2, categories, fontsize=12)
plt.yticks(np.arange(0, 101, 10), fontsize=12)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()