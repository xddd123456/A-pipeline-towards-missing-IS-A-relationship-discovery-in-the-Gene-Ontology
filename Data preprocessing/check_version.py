import pandas as pd

# 读取 2022 和 2023 的 IS_A_relation.csv 文件
file_2022 = "../data/go_2022/is_a_relations.csv"
file_2023 = "../data/go_2025/is_a_relations.csv"

# 用 pandas 读取文件，假设文件是制表符分隔的
data_2022 = pd.read_csv(file_2022, sep='\t')
data_2023 = pd.read_csv(file_2023, sep='\t')

# 提取 2022 年文件中的所有节点集合（包括 id 和 related_id）
nodes_2022 = set(data_2022['id']).union(set(data_2022['related_id']))

# 提取 (id, related_id) 配对并转换为集合
pairs_2022 = set(zip(data_2022['id'], data_2022['related_id']))
pairs_2023 = set(zip(data_2023['id'], data_2023['related_id']))

# 找到 2023 新增的配对
new_pairs = pairs_2023 - pairs_2022

# 筛选新增配对，要求 id 和 related_id 都在 2022 年的节点集合中
filtered_new_pairs = [(id_, related_id) for id_, related_id in new_pairs if id_ in nodes_2022 and related_id in nodes_2022]

# 输出新增配对的数量和部分示例
print(f"满足条件的新增 id-Pair 数量: {len(filtered_new_pairs)}")
print("新增的 id-Pair 示例:", filtered_new_pairs[:10])

# 保存筛选后的配对到文件
output_file = "new_go_pairs_2022_2025.csv"
with open(output_file, "w") as f:
    f.write("id\trelated_id\n")  # 写入表头
    for pair in filtered_new_pairs:
        f.write(f"{pair[0]}\t{pair[1]}\n")

print(f"满足条件的 GO id-Pair 已保存到 {output_file}")
