import pandas as pd

# Step 1: 读取两个文件，只保留前两列
# file2 = "top2_GO_id_to_GO.csv"
# file1 ="common_lines_ndr_100_right+.csv"
# file2 = "pairs_predictions_1_to_GO.csv"
with open('match_counts_summary.csv', 'w') as log_file:
    log_file.write('iteration,matched_row_count\n')  # 写表头
    for i in range(1, 1001):

        file1 = f'../model_prediction/prediction_data/go_2022/fillter/example/combined_top_iter{i}_pre.csv'
        # file1 = './Result/go_2023/one_tree/ndr_pairs_with_uncle_nephew_iter2_top.csv'
        # file1 = 'common_lines_ndr_100.csv'
        # file2 = 'data/go_2024/is_a_relations.csv'
        file2 = "new_go_pairs_2023.csv"
        # file3 = './Result/go_2025/top2_GO_id_to_GO.csv'
        # file3 = 'top2_GO_id_to_GO.csv'
        # file3 = './Result/go_2022/data/combined_top.csv'
        df1 = pd.read_csv(file1, header=None, usecols=[0, 1], sep='\t')
        df2 = pd.read_csv(file2, header=None, usecols=[0, 1], sep='\t', skiprows=1)
        # df3 = pd.read_csv(file3, header=None, usecols=[0, 1], sep='\t')
        # Step 2: 找出前两列相同的部分
        # 通过 inner join 获取两者前两列完全相同的行

        common = pd.merge(df1, df2, how='inner', on=[0, 1])
        # common = pd.merge(df1, df3, how='inner', on=[0, 1])
        # Step 3: 保存结果为新的 CSV 文件
        # output_file = "./Result/go_2023/common_lines_ndr_pairs_with_uncle_nephew_1_new.csv"
        # output_file = "./Result/go_2024/combined_top_new.csv"
        output_file = f'../model_prediction/prediction_data/go_2022/fillter/example/combined_top_iter{i}_pre_true.csv'
        common.to_csv(output_file, index=False, header=False, sep='\t')
        matched_count = len(common)
        log_file.write(f'{i},{matched_count}\n')
        print(f"前两列相同的行已保存到 {output_file}")


