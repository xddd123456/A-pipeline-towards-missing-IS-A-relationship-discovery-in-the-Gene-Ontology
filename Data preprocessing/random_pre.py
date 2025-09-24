import csv
import random

# **Step 1: 读取 is_a_relations.csv，构建 IS-A 关系集合**
is_a_pairs = set()
all_nodes = set()

with open("../data/go_2022/is_a_relations.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        parent, child = row[0], row[1]  # 直接读取字符串
        is_a_pairs.add((parent, child))
        all_nodes.update([parent, child])

# **Step 2: 读取 new_go_2023.csv，存入集合**
new_go_2023_pairs = set()

with open("../Data preprocessing/new_go_pairs_2023.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        parent, child = row[0], row[1]
        new_go_2023_pairs.add((parent, child))

# **Step 3: 进行 100000 次抽样，每次生成 532 个负样本，并记录重复数量**
total_duplicates = 0
output_file = "../data/go_2023/negative_sample_duplicates_per_iteration.csv"
log_file = "../data/go_2023/negative_sample_log.txt"  # 日志文件
with open(output_file, "w", newline='') as csvfile, open(log_file, "w", encoding='utf-8') as log:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(["Iteration", "Duplicate_Count"])  # 写入表头

    all_nodes_list = list(all_nodes)  # 转换为列表，便于随机抽样

    for iteration in range(1, 100001):  # 100000 次抽样
        negative_samples = set()

        while len(negative_samples) < 518:
            node_a, node_b = random.sample(all_nodes_list, 2)  # 直接随机抽取两个不同的 GO ID

            # **确保样本不在 is_a_pairs 中**
            if (node_a, node_b) not in is_a_pairs and node_a != node_b:
                negative_samples.add((node_a, node_b))

        # **计算当前 501 个负样本在 new_go_2023.csv 中的重复数量**
        duplicates_in_iteration = sum(1 for pair in negative_samples if pair in new_go_2023_pairs)
        total_duplicates += duplicates_in_iteration

        # **写入当前迭代的重复数量**
        writer.writerow([iteration, duplicates_in_iteration])
        log_entry = f"完成 {iteration}/100000 次抽样，当前轮重复对数: {duplicates_in_iteration}, 累计重复: {total_duplicates}\n"
        log.write(log_entry)
        # **每 1000 次打印一次，避免大量输出降低性能**
        if iteration % 1000 == 0:
            print(log_entry.strip())
    # **最后写入总重复数**
    writer.writerow(["Total", total_duplicates])
    log.write(f"最终累计重复数量: {total_duplicates}\n")
print(f"重复数量数据已保存到 {output_file}")
print(f"日志已保存到 {log_file}")
