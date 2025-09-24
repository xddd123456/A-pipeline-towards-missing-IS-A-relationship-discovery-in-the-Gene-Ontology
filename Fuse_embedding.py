import torch
import torch.nn as nn
import numpy as np

# 加载嵌入数据
embedding_name = np.load('data_init/go_term_embeddings_N_reshaped.npy')
embedding_def = np.load('data_init/go_2024/go_term_def_embeddings.npy')
# embedding_relations = np.load('data_init/go_substree_embeddings.npy')
embedding_relations = np.load('data_init/GO_other_embeddings.npy')
# 假设已加载的嵌入
embedding_name = torch.tensor(embedding_name, dtype=torch.float32)
embedding_def = torch.tensor(embedding_def, dtype=torch.float32)
embedding_relations = torch.tensor(embedding_relations, dtype=torch.float32)

print("Embedding shapes:", embedding_name.shape)
print("Embedding shapes:", embedding_def.shape)
print("Embedding shapes:", embedding_relations.shape)

# 对 embedding_def 添加一个额外的维度，变成形状为 (47903, 1, 768)
embedding_def = embedding_def.unsqueeze(1)
embedding_relations = embedding_relations.unsqueeze(1)
# 合并这三个嵌入
combined_embedding = torch.cat([embedding_name, embedding_def, embedding_relations], dim=1)

# 打印合并后的形状
print("Combined embedding shape:", combined_embedding.shape)

# 将 PyTorch 张量转换为 NumPy 数组
combined_embedding_numpy = combined_embedding.numpy()

# 保存为 .npy 文件
np.save('data_init/my_embedding.npy', combined_embedding_numpy)

# 输出保存路径
print("Combined embedding saved to 'data_init/my_embedding.npy'")

# 定义卷积降维网络
class ConvDimReduction(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvDimReduction, self).__init__()
        # 使用1D卷积层，将每个样本的特征维度压缩到output_channels
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        # x的形状是 (batch_size, seq_len, input_channels)，即 (47903, 30, 768)
        # 转换为卷积所需的形状 (batch_size, input_channels, seq_len)，即 (47903, 768, 30)
        x = x.permute(0, 2, 1)  # 转置维度
        x = self.conv1(x)  # 卷积操作，输出 (batch_size, output_channels, seq_len)
        x = x.mean(dim=2)  # 对序列维度 (seq_len) 求平均，得到 (batch_size, output_channels)
        return x

# 初始化卷积降维模型
conv_model = ConvDimReduction(input_channels=768, output_channels=128)

# 将合并后的embedding传入卷积模型
reduced_embedding_conv = conv_model(combined_embedding)
print("Reduced Embedding Shape:", reduced_embedding_conv.shape)
reduced_embedding_numpy = reduced_embedding_conv.detach().cpu().numpy()

# 保存为 .npy 文件
np.save('data_init/my_embedding_128.npy', reduced_embedding_numpy)

print("Reduced embedding saved to 'data_init/my_embedding_128.npy'")