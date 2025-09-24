import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置字体更美观
rcParams['font.family'] = 'DejaVu Sans'

# 指标标签
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AUPR']
num_labels = len(labels)

# 数据
# 2022年
# model_A = [0.93456, 0.86754, 0.87382, 0.86167, 0.97633, 0.93645]
# model_B = [0.90948, 0.81841, 0.81667, 0.82109,0.96148, 0.89591]
# 2023年
# model_A = [0.93367, 0.86504, 0.87947, 0.85208, 0.97639, 0.93629]
# model_B = [0.90633, 0.81175, 0.81632, 0.80976, 0.96025, 0.89417]

# 2024年
# model_A = [0.91346, 0.81939, 0.85599, 0.78590, 0.96348, 0.90053]
# model_B = [0.88379, 0.76578, 0.76955, 0.76392, 0.93974, 0.84188]
# 2025年
model_A = [0.90073, 0.79197, 0.82894, 0.76033, 0.95409, 0.87959]
model_B = [0.87766, 0.74943, 0.76359, 0.73613, 0.93210, 0.82493]
models = [model_A, model_B]
model_names = ['With Attention', 'Without Attention']

# 柔和配色 + marker
# colors = ['#CDC1FF', '#FFCCEA']
colors = ['#a6cee3', '#b2df8a']
markers = ['o', 's']
linestyles = ['solid'] * 4
# 角度计算
# 角度计算
angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

# 创建图形
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 画每条线
for i, model in enumerate(models):
    data = model + [model[0]]  # 闭合，不要修改原始 model_A / model_B
    ax.plot(angles, data, color=colors[i], linewidth=2, linestyle=linestyles[i],
            marker=markers[i], label=model_names[i])
    ax.fill(angles, data, color=colors[i], alpha=0.25)

# 添加标签
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)

# 设置坐标范围
ax.set_ylim(0.5, 1)

# 添加图例
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()
