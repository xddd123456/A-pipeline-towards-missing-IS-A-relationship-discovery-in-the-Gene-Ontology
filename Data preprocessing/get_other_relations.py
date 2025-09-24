import json
import csv

# 1. 加载 id_to_index 对应关系（JSON 格式）
with open("../data/go_2020/id_to_index.json", "r") as json_file:
    id_to_index_dict = json.load(json_file)  # 解析 JSON 文件为字典

# 2. 读取 merge_relations.tsv 文件
input_file = "data/go_2020/merged_relations.tsv"
output_file = "../data/go_2020/relations_indexed.tsv"

# 存储结果
indexed_relations = []

# 逐行读取文件并转换
with open(input_file, "r") as infile:
    tsv_reader = csv.reader(infile, delimiter='\t')  # 读取 TSV 文件
    for row in tsv_reader:
        if len(row) >= 2:  # 确保至少有子节点和父节点
            go1, go2 = row[:2]  # 只取前两列，忽略 relation

            # 获取序号（若 ID 不存在于 id_to_index，则跳过）
            index1 = id_to_index_dict.get(go1, None)
            index2 = id_to_index_dict.get(go2, None)

            if index1 is not None and index2 is not None:
                # 保存替换后的序号
                indexed_relations.append(f"{index1}\t{index2}\n")

# 3. 写入结果文件
with open(output_file, "w") as outfile:
    outfile.writelines(indexed_relations)

print(f"转换完成！结果已保存至 {output_file}")
