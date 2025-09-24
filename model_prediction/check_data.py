import pandas as pd

# 读取第一个文件，包含 is_a 关系
# file_1 = pd.read_csv('prediction_data/go_2022/pairs_predictions_1_to_GO.csv', sep='\t', header=None, names=['id', 'related_id', 'relation'])

# 读取第二个文件，包含 id 和 related_id
# file_1 = pd.read_csv('../Data preprocessing/new_go_pairs_2023.csv', sep='\t', header=None, names=['id', 'related_id'])
file_1 = pd.read_csv('../data/go_2022/is_a_relations.csv', sep='\t', header=None, names=['id', 'related_id', 'relation'])
file_2 = pd.read_csv('prediction_data/go_2022/ndr_pairs.csv', sep='\t', skiprows=1,header=None, names=['id', 'related_id'])
# 提取文件1中的 (id, related_id) 对
is_a_relations_1 = set(zip(file_1['id'], file_1['related_id']))

# 提取文件2中的 (id, related_id) 对
relations_2 = set(zip(file_2['id'], file_2['related_id']))

# 找出两个文件中相同的 (id, related_id) 对
common_relations = is_a_relations_1.intersection(relations_2)

# 输出相同的部分
print("相同的 (id, related_id) 对：")
for relation in common_relations:
    print(relation)
# 输出相同对的数量
print(f"相同的 (id, related_id) 对的数量: {len(common_relations)}")