import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入 tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
kf = KFold(n_splits=10, shuffle=True, random_state=42)

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

embedding_matrix = np.load('data/pre_train_embeddings/go_term_embeddings_word_27_64.npy')

# 将 embedding_matrix 转化为 PyTorch Tensor
embedding_tensor = torch.from_numpy(embedding_matrix).float()  # 转换为 float 类型的 Tensor
embedding_tensor = embedding_tensor.reshape(47903, -1)
# 如果需要检查 Tensor 的形
print(embedding_tensor.shape)
name = 'train_1_3_1.csv'
# 读取数据
data = pd.read_csv(name, sep='\t', header=None, names=["X1", "X2", "Label"])

# # 划分训练集和测试集
# train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
#
# x1_data = train_data['X1'].values  # 将 pandas Series 转换为 numpy 数组
# x2_data = train_data['X2'].values  # 将 pandas Series 转换为 numpy 数组
# y_train = train_data['Label'].values  # 标签
#
# x1_test = test_data['X1'].values  # 测试集输入
# x2_test = test_data['X2'].values  # 测试集输入
# y_test = test_data['Label'].values  # 测试集标签


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
# train_loader = create_dataloader(x1_data, x2_data, y_train, batch_size=32, shuffle=True)
# test_loader = create_dataloader(x1_test, x2_test, y_test, batch_size=32, shuffle=False)


import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        query = self.query(x)  # [batch_size, seq_len, input_dim]
        key = self.key(x)  # [batch_size, seq_len, input_dim]
        value = self.value(x)  # [batch_size, seq_len, input_dim]

        # 注意力权重计算
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # 加权求和
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, seq_len, input_dim]
        return attention_output
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None):
        super(CNN, self).__init__()

        # 预训练嵌入层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        # 注意力机制
        self.attention = SelfAttention(input_dim=256)

        # 定义全连接层
        # 您需要根据实际的卷积层输出尺寸来调整这里的输入尺寸
        self.fc1 = nn.Linear(256 * 27, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = x1.to(device)
        x2 = x2.to(device)
        # 获取嵌入
        x1_embedding = self.embedding(x1)
        x2_embedding = self.embedding(x2)

        # 拼接 x1 和 x2 的嵌入向量
        x = x2_embedding - x1_embedding
        x = x.reshape(-1, 27, 64)
        x = x.permute(0, 2, 1)
        # 通过卷积层
        x = self.conv1(x)
        x = F.relu(x)  # ReLU 激活函数
        # x = self.conv2(x)
        # x = F.relu(x)  # ReLU 激活函数
        x = self.conv2(x)
        x = F.relu(x)  # ReLU 激活函数


        # 展平卷积层的输出
        x = x.view(x.size(0), -1)  # 展平成 [batch_size, 256 * embedding_dim]，根据卷积输出的维度调整

        # 全连接层
        x = F.relu(self.fc1(x))  # 全连接层激活
        x = self.dropout(x)
        x = self.fc2(x)  # 最终输出层
        x = F.relu(x)
        x = self.fc3(x)
        # out = self.sigmoid(x)  # Sigmoid 激活函数，适用于二分类问题
        return x

def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    for x1, x2, label in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        # 训练模型
        outputs = model(x1, x2)
        label = label.to(device)
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
            label = label.to(device)
            loss = criterion(outputs, label)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # 计算验证集的指标
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_preds)
    print(
        f"\nEvaluate Loss: {total_loss / len(train_loader):.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n ")



epoch_accuracies = []
epoch_f1_scores = []
epoch_precisions = []
epoch_recalls = []
for fold, (train_index, val_index) in enumerate(kf.split(data)):
    print(f"Fold {fold + 1}")

    # 分割数据
    train_data, val_data = data.iloc[train_index], data.iloc[val_index]
    x1_train, x2_train, y_train = train_data['X1'].values, train_data['X2'].values, train_data['Label'].values
    x1_val, x2_val, y_val = val_data['X1'].values, val_data['X2'].values, val_data['Label'].values

    # 创建 DataLoader
    train_loader = create_dataloader(x1_train, x2_train, y_train, batch_size=256, shuffle=True)
    val_loader = create_dataloader(x1_val, x2_val, y_val, batch_size=256, shuffle=False)

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CNN(vocab_size=47903, embedding_dim=128, pretrained_embeddings=embedding_tensor).to(device)

    model = CNN(vocab_size=47903, embedding_dim=128, pretrained_embeddings=embedding_tensor).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.device = device

    epochs = 10
    # 训练模型
    for epoch in range(epochs):
        train_epoch(train_loader, model, criterion, optimizer)
        evaluate_epoch(val_loader, model, criterion)

    # 在验证集上评估模型并记录指标
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x1, x2, label in val_loader:
            outputs = model(x1.to(device), x2.to(device))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.numpy())

    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_preds)
    epoch_accuracies.append(accuracy)
    epoch_f1_scores.append(f1)
    epoch_precisions.append(precision)
    epoch_recalls.append(recall)

# 输出平均指标
print(f"Average Accuracy: {np.mean(epoch_accuracies)}")
print(f"Average F1 Score: {np.mean(epoch_f1_scores)}")
print(f"Average Precision: {np.mean(epoch_precisions)}")
print(f"Average Recall: {np.mean(epoch_recalls)}")
print(name)

