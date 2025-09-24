import csv
import random

# Step 1: 读取 is_a_relations_indexed.csv 文件并构建 IS-A 关系集合
is_a_pairs = set()
all_nodes = set()

with open("../data/go_2025/is_a_relations_indexed.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    next(reader)  # 跳过表头
    for row in reader:
        parent = int(row[0])
        child = int(row[1])
        is_a_pairs.add((parent, child))
        all_nodes.update([parent, child])

# Step 2: 生成负样本
negative_samples = set()
num_negative_samples = len(is_a_pairs) * 5  # 生成与正样本数量相同的负样本

while len(negative_samples) < num_negative_samples:
    # 随机选择两个节点生成负样本对
    node_a, node_b = random.sample(list(all_nodes), 2)

    # 确保该对不存在于 is_a_pairs 中，并且不是同一节点
    if (node_a, node_b) not in is_a_pairs and node_a != node_b:
        negative_samples.add((node_a, node_b, 0))

# Step 3: 保存负样本到 CSV 文件
with open("../data/go_2025/is_a_relations_negative_samples_5.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(["parent_id", "child_id", "relation"])  # 表头
    writer.writerows(negative_samples)

print("负样本已生成并保存到 is_a_relations_negative_samples_5.csv 文件中。")
