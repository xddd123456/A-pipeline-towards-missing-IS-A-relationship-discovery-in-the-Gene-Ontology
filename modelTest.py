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
embedding_matrix = np.load('data/pre_train_embeddings/fused_embedding_nn.npy')
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


# 自定义 Dataset 类来加载特征和标签
class TextPairDataset(Dataset):
    def __init__(self, data, embedding_matrix):
        self.data = data
        self.embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)  # 转化为张量
        # self.embedding_matrix = embedding_matrix
        self.text_A = torch.tensor(data['X1'].values, dtype=torch.long)  # 假设 X1 是序号
        self.text_B = torch.tensor(data['X2'].values, dtype=torch.long)  # 假设 X2 是序号
        self.labels = torch.tensor(data['Label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x1_embedding = self.embedding_matrix[self.text_A[idx]]
        x2_embedding = self.embedding_matrix[self.text_B[idx]]
        label = self.labels[idx]
        return x1_embedding, x2_embedding, label


def create_dataloader(data, embedding_matrix, batch_size=32, shuffle=True):
    dataset = TextPairDataset(data, embedding_matrix)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# 示例
train_loader = create_dataloader(train_data, embedding_matrix, batch_size=32, shuffle=True)
test_loader = create_dataloader(test_data, embedding_matrix, batch_size=32, shuffle=False)


# 定义 BiLSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        # self.fc1 = nn.Linear(hidden_dim * 2, 128)
        # self.fc2 = nn.Linear(128, 32)
        # self.fc3 = nn.Linear(32, output_dim)

        # self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, 64)
        # self.fc3 = nn.Linear(64, 16)
        # self.fc4 = nn.Linear(16, output_dim)
        # self.output = nn.Linear(output_dim, 2)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x1, x2):
        x1_embedding = self.embedding(x1)  # 获取 x1 的嵌入表示
        x2_embedding = self.embedding(x2)  # 获取 x2 的嵌入表示
        x = torch.cat((x1, x2), dim=1).unsqueeze(1)  # (batch_size, 1, embedding_dim * 2)

        # LSTM 层
        out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        out = out[:, -1, :]  # 取最后一个时间步的输出

        out = self.dropout(out)
        # 全连接层和 ReLU 激活
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        # 最后一层输出
        out = self.fc4(out)

        # 使用 softmax 激活函数输出二分类的概率
        out = F.softmax(out, dim=1)  # softmax 对应多分类问题

        return out


# weights = torch.tensor([1.0, 3.0])
# 初始化模型、损失函数和优化器
model = BiLSTM(embedding_dim=768, hidden_dim=256, output_dim=1)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.004)


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    for x1_embedding, x2_embedding, label in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        # 在这里传入 x1 和 x2
        outputs = model(x1_embedding, x2_embedding)
        loss = criterion(outputs, label.long())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 获取预测值并转换为类标签
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        # 计算训练集的指标
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_preds)
    # accuracy_train = accuracy_score(all_labels, all_preds)
    # precision_train = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    # recall_train = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    # f1_train = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    print(
        f"\nTrain Loss: {total_loss / len(train_loader):.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")


def evaluate_epoch(val_loader, model, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for x1_embedding, x2_embedding, label in tqdm(val_loader, desc="Evaluating"):
            outputs = model(x1_embedding, x2_embedding)
            loss = criterion(outputs, label.long())
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # # 计算验证集的指标
    # accuracy_val = accuracy_score(all_labels, all_preds)
    # precision_val = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    # recall_val = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    # f1_val = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_preds)
    print(
        f"\nValidation Loss: {total_loss / len(val_loader):.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")


# 训练和验证循环
epochs = 100
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_epoch(train_loader, model, criterion, optimizer)
    evaluate_epoch(test_loader, model, criterion)
