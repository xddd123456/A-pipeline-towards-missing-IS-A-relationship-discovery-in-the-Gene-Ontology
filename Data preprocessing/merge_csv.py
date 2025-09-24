# import pandas as pd
#
# # 手动指定列名，如果你知道具体列数
# column_names = ["col1", "col2", "col3"]
#
# # 读取 CSV 文件并指定列数，防止多余逗号影响
# df1 = pd.read_csv("is_a_relations_indexed.csv", header=None, names=column_names, usecols=[0, 1, 2], delimiter='\t')
# df2 = pd.read_csv("is_a_relations_negative_samples_3.csv", header=None, names=column_names, usecols=[0, 1, 2], delimiter='\t')
# df3 = pd.read_csv("uncle_nephew_relations_clean.csv", header=None, names=column_names, usecols=[0, 1, 2], delimiter='\t')
# # df2_filtered = df2.sample(frac=0.5, random_state=42)
# # df3_filtered = df3.sample(frac=0.5, random_state=42)
# # 合并数据
# merged_df = pd.concat([df1, df2], ignore_index=True)
#
# # 保存合并后的文件，不包含表头
# merged_df.to_csv("train.csv", index=False, header=False, sep='\t')
#
# print("文件已合并为 train.csv，且无多余的逗号。")


# 获取同等长度1：1：1
import pandas as pd

# 手动指定列名
column_names = ["col1", "col2", "col3"]

# 读取 CSV 文件
df1 = pd.read_csv("../data/go_2025/is_a_relations_indexed.csv", header=None, names=column_names, usecols=[0, 1, 2], skiprows=1, delimiter='\t')
df2 = pd.read_csv("../data/go_2025/is_a_relations_negative_samples_5.csv", header=None, names=column_names, usecols=[0, 1, 2], skiprows=1, delimiter='\t')
df3 = pd.read_csv("../data/go_2025/uncle_nephew_relations.csv", header=None, names=column_names, usecols=[0, 1, 2], skiprows=1, delimiter='\t')
# 获取df1的长度
df1_length = len(df1)
print(df1_length)
# 同样处理df3
if len(df2) > df1_length:
   df2 = df2.sample(n=df1_length, random_state=42)
if len(df3) > df1_length:
   df3 = df3.sample(n=df1_length*2, random_state=42)

# 合并数据
merged_df = pd.concat([df1, df2, df3], ignore_index=True)
# merged_df = pd.concat([df1, df3], ignore_index=True)
# 保存合并后的文件，不包含表头
merged_df.to_csv("../data/go_2025/train_2.csv", index=False, header=False, sep='\t')

print("文件已合并为 train.csv")