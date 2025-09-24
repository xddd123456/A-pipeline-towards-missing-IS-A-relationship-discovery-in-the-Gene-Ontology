import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import json

MAX_SEQ_LEN = 27  # 最大词数

def get_padded_embeddings(text, model, tokenizer, device):
    """
    获取文本的 BERT 嵌入，并填充或截断到固定长度 MAX_SEQ_LEN。
    """
    # 对输入文本进行分词和编码
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 获取最后一层的所有 token 的嵌入
    last_hidden_state = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (seq_len, 768)

    # 获取原始分词后的 token
    tokens = tokenizer.tokenize(text)

    # 去掉特殊标记 [CLS] 和 [SEP] 的嵌入
    token_embeddings = last_hidden_state[1:-1]  # 去掉 [CLS] 和 [SEP]

    # 填充或截断到固定长度 MAX_SEQ_LEN
    if len(token_embeddings) > MAX_SEQ_LEN:
        token_embeddings = token_embeddings[:MAX_SEQ_LEN]  # 截断
    else:
        padding = np.zeros((MAX_SEQ_LEN - len(token_embeddings), 768))  # 填充 0 向量
        token_embeddings = np.vstack((token_embeddings, padding))  # 拼接

    return token_embeddings

def process_go_terms(input_file, output_file, model, tokenizer, device):
    """
    处理 GO terms，生成固定大小的嵌入矩阵，并保存到文件。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        go_terms = json.load(f)

    embeddings = np.zeros((len(go_terms), MAX_SEQ_LEN, 768))  # 初始化最终的嵌入矩阵

    for i, go_term in enumerate(go_terms):
        name = go_term["name"]
        token_embeddings = get_padded_embeddings(name, model, tokenizer, device)
        embeddings[i] = token_embeddings  # 将结果填入对应位置

        if i % 1000 == 0:  # 打印进度
            print(f"Processed {i}/{len(go_terms)} GO terms.")

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
output_file = "data_init/go_2025/go_name_embeddings"  # 输出文件名（不包含扩展名，默认为 .npy）

# 处理 GO terms 并提取 name 嵌入
process_go_terms(input_file, output_file, model, tokenizer, device)
