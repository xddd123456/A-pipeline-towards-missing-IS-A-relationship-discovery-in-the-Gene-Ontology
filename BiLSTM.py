import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入 tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F


def triplet_loss(anchor, positive, negative, margin_constant):
    negative_dis = torch.sum(anchor * negative, dim=1)
    positive_dis = torch.sum(anchor * positive, dim=1)
    margin = margin_constant * torch.ones(negative_dis.shape).cuda()
    diff_dis = negative_dis - positive_dis
    penalty = diff_dis + margin
    triplet_loss = 1 * torch.max(penalty, torch.zeros(negative_dis.shape).cuda())
    return torch.mean(triplet_loss)

def calculate_metrics(y_true, y_pred):
    # 将预测结果转换为整数类型
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    y_true = y_true.to(torch.long)
    y_pred = y_pred.to(torch.long)

    # 计算混淆矩阵
    confusion_matrix = torch.zeros((2, 2), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        confusion_matrix[t, p] += 1

    # 从混淆矩阵中提取值
    tp = confusion_matrix[1, 1]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]
    tn = confusion_matrix[0, 0]

    # 计算准确率
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    # 计算精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # 计算召回率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, f1, precision, recall


# 假设你的预训练嵌入矩阵是一个 numpy 数组
# embedding_matrix = np.load('data/pre_train_embeddings/fused_embedding_weighted.npy')  # shape: (47903, 768)
# embedding_matrix = np.load('go_term_name_embeddings.npy')  # shape: (47903, 768)


# embedding_matrix = np.load('data/pre_train_embeddings/go_term_embedding_128.npy')

# bert 27*768
embedding_matrix = np.load('data/pre_train_embeddings/go_term_embeddings_N_reshaped_128.npy')

# 将 embedding_matrix 转化为 PyTorch Tensor
embedding_tensor = torch.from_numpy(embedding_matrix).float()  # 转换为 float 类型的 Tensor

# 如果需要检查 Tensor 的形状
print(embedding_tensor.shape)


class EmbeddingFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingFusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# embedding_matrix = np.load('data/pre_train_embeddings/fused_embedding_concat.npy')  # shape: (47903, 768*3)
# model = EmbeddingFusion(768 * 3, 768)

# embedding_matrix = model(torch.tensor(embedding_matrix, dtype=torch.float32)).detach().numpy()

# 读取数据
data = pd.read_csv('train.csv', sep='\t', header=None, names=["X1", "X2", "Label"])

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

x1_data = train_data['X1'].values  # 将 pandas Series 转换为 numpy 数组
x2_data = train_data['X2'].values  # 将 pandas Series 转换为 numpy 数组
y_train = train_data['Label'].values  # 标签

x1_test = test_data['X1'].values  # 测试集输入
x2_test = test_data['X2'].values  # 测试集输入
y_test = test_data['Label'].values  # 测试集标签


# 自定义 Dataset 类来加载特征和标签
class TextPairDataset(Dataset):
    def __init__(self, x1_data, x2_data, labels):
        if isinstance(x1_data, pd.Series):
            x1_data = x1_data.to_numpy()  # 或者 x1_data.values
        if isinstance(x2_data, pd.Series):
            x2_data = x2_data.to_numpy()  # 或者 x2_data.values
        if isinstance(labels, pd.Series):
            labels = labels.to_numpy()

        self.x1_data = torch.tensor(x1_data, dtype=torch.long)  # 将 numpy.ndarray 转为 Tensor
        self.x2_data = torch.tensor(x2_data, dtype=torch.long)  # 同理
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.x1_data)

    def __getitem__(self, idx):
        x1 = self.x1_data[idx]  # 获取 x1
        x2 = self.x2_data[idx]  # 获取 x2
        label = self.labels[idx]  # 获取标签
        return x1, x2, label


def create_dataloader(x1_data, x2_data, labels, batch_size=32, shuffle=True):
    # 构造自定义的 Dataset
    dataset = TextPairDataset(x1_data, x2_data, labels)
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# 示例
train_loader = create_dataloader(x1_data, x2_data, y_train, batch_size=32, shuffle=True)
test_loader = create_dataloader(x1_test, x2_test, y_test, batch_size=32, shuffle=False)


# 定义 BiLSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_embeddings):
        super(BiLSTM, self).__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        # 双向 LSTM 输出的维度是 hidden_dim * 2
        self.fc1 = nn.Linear(hidden_dim * 2, 8)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        #self.fc4 = nn.Linear(16, 2)  # 二分类输出

    def forward(self, x1, x2):
        # 获取 x1 和 x2 的嵌入表示
        x1_embedding = self.embedding(x1)
        x2_embedding = self.embedding(x2)

        # 拼接 x1 和 x2 的嵌入向量
        x = (x2_embedding - x1_embedding).unsqueeze(1)

        # 第一个 LSTM 层
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        # # 第二个 LSTM 层
        # out, _ = self.lstm2(out)
        # out = self.dropout2(out)

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # # Dropout 层
        # out = self.dropout(out)

        # 全连接层和 ReLU 激活
        out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))

        # 最后一层输出
        out = self.fc3(out)

        # 输出为 logits（CrossEntropyLoss 会自动处理 softmax）
        return out


# 初始化 BiLSTM 模型
model = BiLSTM(vocab_size=47903, embedding_dim=128, hidden_dim=32, pretrained_embeddings=embedding_tensor)

# 损失函数使用 CrossEntropyLoss，用于二分类任务
criterion = nn.CrossEntropyLoss()

# 优化器使用 Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    for x1, x2, label in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        # 训练模型
        outputs = model(x1, x2)

        # 计算损失
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 获取预测值并转换为类标签
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    # 计算训练集的指标
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_preds)

    print(
        f"\nTrain Loss: {total_loss / len(train_loader):.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n")


def evaluate_epoch(val_loader, model, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for x1, x2, label in tqdm(val_loader, desc="Evaluating"):
            outputs = model(x1, x2)
            loss = criterion(outputs, label)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # 计算验证集的指标
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_preds)
    print(
        f"\nEvaluate Loss: {total_loss / len(train_loader):.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n ")


# 训练和验证循环
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_epoch(train_loader, model, criterion, optimizer)
    evaluate_epoch(test_loader, model, criterion)
