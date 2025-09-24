import pandas as pd
import json


def load_go_terms(json_file):
    """加载GO术语及其父子关系."""
    with open(json_file, 'r', encoding='utf-8') as file:
        go_data = json.load(file)
    go_dict = {}  # GO ID到名称的映射
    go_parents = {}  # GO ID到直接父ID列表的映射
    for entry in go_data:
        go_id = entry['id']
        go_name = entry['name']
        parents = entry.get('parents', [])  # 假设父术语字段为'parents'
        go_dict[go_id] = go_name
        go_parents[go_id] = parents
    return go_dict, go_parents


def convert_go_pairs_with_relation(ndr_file, go_dict, go_parents, output_file):
    """转换GO ID为名称，并仅保留直接父子关系的对."""
    # 读取NDR Pairs文件
    ndr_df = pd.read_csv(ndr_file, sep='\t', header=None, names=['GO_1', 'GO_2'])

    # 过滤并格式化直接父子关系对
    formatted_lines = []
    for _, row in ndr_df.iterrows():
        go1_id = row['GO_1']
        go2_id = row['GO_2']

        # 获取名称（若不存在则保留ID）
        go1_name = go_dict.get(go1_id, go1_id)
        go2_name = go_dict.get(go2_id, go2_id)
        formatted_lines.append(f"Does “{go1_name}” have an IS_A relationship with “{go2_name}”? Please answer YES or NO.")

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(formatted_lines))
    print(f"转换完成，结果已保存至 {output_file}")


# 文件路径
ndr_file = 'prediction_data/go_2025/one_tree/ndr_pairs_with_uncle_nephew_iter1_pre.csv'
go_term_file = '../data/go_2025/go_terms.json'
output_file = 'prediction_data/go_2025/one_tree/ndr_pairs_with_uncle_nephew_iter1_with_names.csv'
# ndr_file = '../Data preprocessing/new_go_pairs_2025.csv'
# go_term_file = '../data/go_2024/go_terms.json'
# output_file = '../Data preprocessing/new_go_pairs_2025_name.csv'


# 加载数据并转换
go_dict, go_parents = load_go_terms(go_term_file)
convert_go_pairs_with_relation(ndr_file, go_dict, go_parents, output_file)