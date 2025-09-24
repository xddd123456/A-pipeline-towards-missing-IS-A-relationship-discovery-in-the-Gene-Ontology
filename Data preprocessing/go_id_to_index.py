import csv
import json

# 初始化字典用于存储编号信息
id_to_index = {}

# 读取 CSV 文件中的 id 列，并为每个 id 分配编号
with open("../go_term_def_embeddings.csv", "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for index, row in enumerate(reader):
        go_id = row["id"]
        id_to_index[go_id] = index

# 将编号后的数据写入 JSON 文件
with open("../data/go_2024/id_to_index.json", "w") as jsonfile:
    json.dump(id_to_index, jsonfile, indent=4)

print("每个 id 已编号并保存到 id_to_index.json")
