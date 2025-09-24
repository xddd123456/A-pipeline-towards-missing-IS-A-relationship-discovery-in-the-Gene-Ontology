import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import from_networkx
import numpy as np
import pickle

# Step 1: 加载 GO 数据
with open('data/go_2024/go_terms.json', 'r') as file:
    go_terms = json.load(file)

# 构建 GO 术语名称和 ID 的字典
go_terms_dict = {term["name"]: term["id"] for term in go_terms}

# Step 2: 加载 id_to_index.json
with open('data/go_2024/id_to_index.json', 'r') as f:
    id_to_index = json.load(f)

# 打印 id_to_index 内容（检查索引是否一致）
for go_id, idx in id_to_index.items():
    print(f"GO ID: {go_id}, Index: {idx}")

# Step 3: 构建子串图
def build_substring_graph(go_terms_dict):
    """
    根据 GO 名称的子串关系构建图，节点是 GO ID，边表示名称的子串关系。
    """
    G = nx.DiGraph()
    names = list(go_terms_dict.keys())
    ids = list(go_terms_dict.values())

    # 添加节点
    for go_id in ids:
        G.add_node(go_id)

    # 添加边（基于子串关系）
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i != j and name1 in name2:  # 如果名称1是名称2的子串
                G.add_edge(ids[i], ids[j])

    return G

# 构建图
G = build_substring_graph(go_terms_dict)

# Step 4: 保存节点顺序（GO_id 和节点索引的映射）
node_order = {go_id: idx for idx, go_id in enumerate(G.nodes)}  # 保存 GO_id 和索引
with open('node_order.json', 'w') as f:
    json.dump(node_order, f, indent=4)

# 转换为 PyTorch Geometric 的数据格式
data = from_networkx(G)

# Step 5: 初始化节点特征（随机初始化768维）
num_nodes = len(data.x) if data.x is not None else len(G.nodes)
data.x = torch.rand((num_nodes, 1536))  # 每个节点初始化为 768 维随机向量

# Step 6: 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 初始化模型
input_dim = 1536
hidden_dim = 1024
output_dim = 2 * 768  # 输出维度为 2 通道 * 768
model = GCN(input_dim, hidden_dim, output_dim)

# Step 7: 训练 GCN 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out, data.x)  # 使用自监督训练方式
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Step 8: 获取训练好的节点 embedding
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 不需要计算梯度
    node_embeddings = model(data.x, data.edge_index)  # 获取节点 embedding

# Step 9: 将 GO ID 和其对应的 embedding 保存到文件

embedding_dict = {}
num_nodes = len(node_embeddings)  # 获取节点总数
output_dim = node_embeddings.size(1)  # 获取每个节点嵌入的维度

# 加载节点顺序
with open('node_order.json', 'r') as f:
    node_order = json.load(f)

# 检查 id_to_index 是否与图节点顺序一致
for go_id, idx in id_to_index.items():
    # 查找节点在图中的位置
    if go_id in node_order:
        node_index = node_order[go_id]
        if node_index < num_nodes:  # 如果索引在有效范围内
            embedding_dict[go_id] = node_embeddings[node_index].cpu().numpy().tolist()  # 转为 numpy 数组再转为列表
        else:  # 如果索引超出范围，填充为 0 向量
            print(f"Warning: Index {node_index} for GO ID {go_id} is out of bounds. Using a zero vector.")
            embedding_dict[go_id] = np.zeros(output_dim).tolist()  # 填充为全零向量
    else:
        # 如果该 GO_id 没有在图中出现，则使用零向量
        print(f"GO ID {go_id} not found in the graph. Using a zero vector.")
        embedding_dict[go_id] = np.zeros(output_dim).tolist()  # 填充为全零向量

# 保存为 JSON 文件
with open('go_id_embeddings.json', 'w') as f:
    json.dump(embedding_dict, f, indent=4)

print("GO ID embeddings 已保存到 'go_id_embeddings.json'")

# 或者使用 Pickle 格式保存
# 使用 Pickle 格式保存
with open('go_id_embeddings.pkl', 'wb') as f:
    pickle.dump(embedding_dict, f)

print("GO ID embeddings 已保存到 'go_id_embeddings.pkl'")
