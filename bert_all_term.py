import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import json

MAX_SEQ_LEN = 27  # 嵌入的最大序列长度

def get_embeddings(text, model, tokenizer, device):
    """
    获取文本的 BERT 嵌入，直接使用文本作为整体输入。
    """
    # 对整个文本进行编码
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding="max_length").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 获取最后一层的 [CLS] 标记嵌入
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # 只取 [CLS] 的向量

    return cls_embedding

def process_go_terms(input_file, output_file, model, tokenizer, device):
    """
    处理 GO terms，生成嵌入矩阵，并保存到文件。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        go_terms = json.load(f)

    embeddings = np.zeros((len(go_terms), 768))  # 初始化最终的嵌入矩阵 (GO_terms 数量, 768)

    for i, go_term in enumerate(go_terms):
        # 使用 GO term 名称
        name = go_term["name"]
        cls_embedding = get_embeddings(name, model, tokenizer, device)
        embeddings[i] = cls_embedding  # 将结果填入对应位置

        if i % 1000 == 0:  # 打印进度
            print(f"Processed {i}/{len(go_terms)} GO terms. Name: {name}")

    # 保存结果为 .npy 文件
    np.save(output_file, embeddings)
    print(f"Embedding 数据已保存到 {output_file}.npy")

# 模型和设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "F:/bert_base_uncare"  # 本地模型路径
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path).to(device)

# 输入和输出文件路径
input_file = "data/go_2025/go_terms.json"  # 包含 GO 条目的文件
output_file = "data_init/go_2025/go_name_embeddings_term"  # 输出文件名（不包含扩展名，默认为 .npy）

# 处理 GO terms 并提取名称嵌入
process_go_terms(input_file, output_file, model, tokenizer, device)
