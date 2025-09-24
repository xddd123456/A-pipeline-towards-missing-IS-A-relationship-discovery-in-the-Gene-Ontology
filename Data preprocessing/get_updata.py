import json
import pandas as pd

# 读取2022年和2023年的 go_terms.json 文件
with open('../data/go_2022/go_terms.json', 'r') as f:
    go_terms_2022 = json.load(f)

with open('../data/go_2025/go_terms.json', 'r') as f:
    go_terms_2023 = json.load(f)

# 提取2022年和2023年中的所有id
go_terms_2022_ids = {term['id'] for term in go_terms_2022}
go_terms_2023_ids = {term['id'] for term in go_terms_2023}

# 找出在2022年和2023年都出现的id
common_ids = go_terms_2022_ids.intersection(go_terms_2023_ids)
# 读取2022年和2023年的 is_a_relation 数据
data_2022 = pd.read_csv('../data/go_2022/is_a_relations.csv', sep='\t')
data_2023 = pd.read_csv('../data/go_2025/is_a_relations.csv', sep='\t')

# 筛选2023年版本中 id 和 related_id 都在公共id集合中的条目

is_a_2023_filtered = data_2023[data_2023['id'].isin(common_ids) & data_2023['related_id'].isin(common_ids)]

# 去除2023年版本中已经存在的条目，只保留2023年中新增的条目
final_relations = is_a_2023_filtered[~is_a_2023_filtered.apply(
    lambda row: ((data_2022['id'] == row['id']) &
                 (data_2022['related_id'] == row['related_id'])).any(), axis=1)]

# 输出结果
print("在2025年版本中新增的 `is_a` 关系条目：")
print(final_relations)

