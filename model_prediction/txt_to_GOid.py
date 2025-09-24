import pandas as pd
import json



# 初始化记录列表
matched_counts = []

# 打开log文件准备写入统计结果
with open("prediction_log.csv", "w") as log_file:
    log_file.write("iteration,positive_prediction_count\n")

    # 文件路径
    for i in range(1, 1001):
        prediction_file = f"prediction_data/go_2022/fillter/example/combined_top_iter{i}_index_pre.txt"  # 替换为你的预测文件路径
        id_to_index_file = "../data/go_2022/id_to_index.json"    # 替换为你的 id_to_index.json 文件路径
        output_file = f"prediction_data/go_2022/fillter/example/combined_top_iter{i}_pre.csv"     # 替换为你的输出文件路径

        # 加载预测文件
        predictions = pd.read_csv(prediction_file, delimiter="\t")
        print(f"预测文件加载完成，数据形状: {predictions.shape}")

        # 加载 id_to_index.json 文件
        with open(id_to_index_file, "r") as f:
            id_to_index = json.load(f)

        # 创建 index_to_id 反向映射
        index_to_id = {str(value): key for key, value in id_to_index.items()}

        # 转换 X1 和 X2 为对应的 GO:ID 格式
        def map_to_go(index):
            return index_to_id.get(str(index), f"Unknown_{index}")

        predictions["GO_X1"] = predictions["X1"].map(map_to_go)
        predictions["GO_X2"] = predictions["X2"].map(map_to_go)

        # 明确复制筛选的结果，确保创建一个新 DataFrame，而不是视图
        predictions_filtered = predictions[predictions["Prediction"] == 1].copy()

        # 接下来就不会有警告
        predictions_filtered["formatted"] = predictions_filtered.apply(
            lambda row: f"{row['GO_X1']}\t{row['GO_X2']}", axis=1
        )

        # 保存到文件
        with open(output_file, "w") as f:
            f.write("\n".join(predictions_filtered["formatted"]))
            # 记录当前文件中 positive 的数量
            count = len(predictions_filtered)
            matched_counts.append(count)
            log_file.write(f"{i},{count}\n")

            print(f"[iter {i}] 预测为1的对数：{count}，保存至：{output_file}")

        # 所有循环完成后，记录平均值
        avg_count = sum(matched_counts) / len(matched_counts)
        log_file.write(f"\nAverage,{avg_count:.2f}\n")
        print(f"\n所有迭代完成，平均预测为1的数量为：{avg_count:.2f}")

