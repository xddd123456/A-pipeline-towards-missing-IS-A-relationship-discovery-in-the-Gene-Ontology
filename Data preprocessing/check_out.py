import pandas as pd

# 读取2022年和2023年的IS_A_relation文件
file_2022 = "../data/go_2022/is_a_relations.csv"
file_2023 = "../data/go_2023/is_a_relations.csv"

# 加载数据，假设文件是以制表符分隔的
data_2022 = pd.read_csv(file_2022, sep='\t')
data_2023 = pd.read_csv(file_2023, sep='\t')

# 提取2022年和2023年的所有GO ID（包括id和related_id）
ids_2022 = set(data_2022['id']).union(set(data_2022['related_id']))
ids_2023 = set(data_2023['id']).union(set(data_2023['related_id']))

# 找到2022年中出现但2023年中没有的GO ID
missing_ids = ids_2022 - ids_2023

# 输出结果
print(f"2022年中出现但2023年中没有的GO ID数量: {len(missing_ids)}")
print("示例缺失的GO ID:", list(missing_ids)[:10])  # 打印部分示例

# 将结果保存到文件
output_file = "missing_go_ids_2022.csv"
with open(output_file, "w") as f:
    f.write("GO_ID\n")  # 写入表头
    for go_id in missing_ids:
        f.write(f"{go_id}\n")

print(f"缺失的GO ID已保存到 {output_file}")
