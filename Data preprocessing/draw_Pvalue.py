import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from collections import Counter

# 读取数据
df = pd.read_csv("../data/go_2023/negative_sample_duplicates_per_iteration.csv", sep='\t', header=None)
labels = df[1].astype(int).values

group_size = 1000
num_groups = 100

# 计算每组验证成功的数量（整数）
counts = [np.sum(labels[i*group_size:(i+1)*group_size]) for i in range(num_groups)]

# 统计每个验证成功数量在100组中出现的频数
count_freq = Counter(counts)
x = sorted(count_freq.keys())                    # 成功数值（0, 1, 2,...）
y = [count_freq[i] for i in x]                   # 每个值的出现次数

# 拟合正态分布
mu = np.mean(counts)
sigma = np.std(counts)
x_fit = np.linspace(min(x)-1, max(x)+1, 200)
y_fit = norm.pdf(x_fit, mu, sigma) * num_groups  # 乘以样本数变频数高度

# 画图
plt.figure(figsize=(10, 6))

# 直方图（其实这里我们是按频数画柱子）
plt.bar(x, y, width=0.4, color='#F27873', edgecolor='black', label='Validation Count per Random Group')

# 拟合曲线
plt.plot(x_fit, y_fit, color='#DAA628', label='Smoothed Distribution')

plt.xlabel("Number of Validated IS-A Pairs per Group")
plt.ylabel("Number of Groups")
plt.title("Distribution of validated IS-A pairs from randomly selected pairs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
mean_count = np.mean(counts)
std_count = np.std(counts)  # 默认为 population std，即除以 n

print(f"平均验证计数（μ）: {mean_count:.3f}")
print(f"标准差（σ）: {std_count:.3f}")