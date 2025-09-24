import pandas as pd

# 读取两个文件
pairs1 = pd.read_csv("prediction_data/go_2022/pairs_predictions_1_to_GO.csv", sep='\t', header=None, names=["Node1", "Node2", "Relation"])
pairs2 = pd.read_csv("../data/go_2023/is_a_relations.csv" , sep='\t', header=None, names=["Node1", "Node2", "Relation"])

# 合并两个表，找出重合的部分
merged = pd.merge(pairs1, pairs2, on=["Node1", "Node2", "Relation"], how="inner")

# 保存重合部分到新文件
output_file = "prediction_data/go_2022/overlapping_pairs_change_2024.csv"
merged.to_csv(output_file, sep='\t', index=False)

print(f"重合部分已保存到 {output_file}")
