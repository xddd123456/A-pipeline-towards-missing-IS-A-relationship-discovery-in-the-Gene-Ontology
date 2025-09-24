import pandas as pd
import json

# 读取 top2_new.csv
top2_df = pd.read_csv('top2_new.csv', header=None)

# 读取 go_terms.json
with open('../data/go_2022/go_terms.json', 'r') as f:
    go_data = json.load(f)

# 创建 GO ID -> Name 的映射
go_dict = {entry['id']: entry['name'] for entry in go_data}

# 替换 GO ID 为对应的 Name
top2_df['name1'] = top2_df[0].map(go_dict)
top2_df['name2'] = top2_df[1].map(go_dict)

top2_df['name1_word_count'] = top2_df['name1'].str.split().str.len()
top2_df['name2_word_count'] = top2_df['name2'].str.split().str.len()

# 导出结果
top2_df[['name1', 'name2', 'name1_word_count', 'name2_word_count']].to_csv('top2_names.csv', index=False)

print("转换完成，结果已保存在 'top2_names.csv'。")
