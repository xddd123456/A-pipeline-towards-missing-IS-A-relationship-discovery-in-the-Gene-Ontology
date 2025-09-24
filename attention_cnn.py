import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 导入 tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
from co_attention import SA
from co_attention import SGA
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F


class TeLU(nn.Module):
    def __init__(self):
        super(TeLU, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.exp(x))
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
    print(tp)
    print(fp)
    print(fn)
    print(tn)
    # 计算准确率
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    # 计算精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # 计算召回率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, f1, precision, recall


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


def create_dataloader(x1_data, x2_data, labels, batch_size=256, shuffle=True):
    # 构造自定义的 Dataset
    dataset = TextPairDataset(x1_data, x2_data, labels)
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader




class ScaledDotProductAttention(nn.Module):
    def __init__(self, input_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.input_dim = input_dim

    def forward(self, query, key, value):
        # query, key, value: [batch_size, seq_len, input_dim]
        # 计算 QK^T / sqrt(d_k) 的点积
        d_k = query.size(-1)  # 获取 key 的最后一维大小（即 embedding 维度）
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # 用注意力权重对 value 进行加权
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
        # LSTM 层
        self.lstm = nn.LSTM(embedding_dim, hidden_size=2, num_layers=2, batch_first=True, bidirectional=True)

        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.SGA = SGA()
        self.telu = TeLU()
        self.SA = SA()
        self.fc1 = nn.Linear(embedding_dim*256, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x1, x2):
        # 获取嵌入
        x1 = x1.to(device)
        x2 = x2.to(device)
        x1_embedding = self.embedding(x1)
        x2_embedding = self.embedding(x2)
        # 拼接 x1 和 x2 的嵌入向量
        x = x2_embedding - x1_embedding
        x = x.unsqueeze(1)  # 增加一个维度，从[batch_size, embedding_dim]变为[batch_size, 1, embedding_dim]
        # x, _ = self.lstm(x)  # x: [batch_size, seq_len, hidden_size*2], _=
        # 通过卷积层
        x = self.conv1(x)
        # x = F.relu(x)
        x = self.telu(x)
        x = self.conv2(x)
        # x = F.relu(x)
        x = self.telu(x)
        x = self.conv3(x)
        # x = F.relu(x)
        x = self.telu(x)
        x = self.SA(x)
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.fc1(x)
        # x = F.relu(x)
        x = self.telu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        out = self.fc3(x)

        return out




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
    return  total_loss / len(train_loader), accuracy, f1, precision, recall


def evaluate_epoch(val_loader, model, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    probabilities = []
    with torch.no_grad():
        for x1, x2, label in tqdm(val_loader, desc="Evaluating"):
            outputs = model(x1, x2)
            label = label.to(device)
            loss = criterion(outputs, label)
            total_loss += loss.item()
            all_probabilities = F.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            probabilities.extend(all_probabilities.cpu().numpy())

    # 计算验证集的指标
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_preds)
    auc_out = roc_auc_score(all_labels, probabilities)
    aupr = average_precision_score(all_labels, probabilities)
    print(
        f"\nEvaluate Loss: {total_loss / len(train_loader):.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}  | AUC: {auc_out:.4f} | AUPR: {aupr:.4f}\n")  #
    return total_loss / len(val_loader), accuracy, f1, precision, recall, auc_out, aupr
4



name = 'data/go_2022/train_2.csv'
# 读取数据
# only random neg
data = pd.read_csv(name, sep='\t', header=None, names=["X1", "X2", "Label"])
# 合并特征
X1 = data["X1"].values.reshape(-1, 1)  # 如果 X1 是单列，需要 reshape 成二维数组
X2 = data["X2"].values.reshape(-1, 1)
Labels = data["Label"].values  # 标签
# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)  # first: 42 second: 32
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)


# Prepare train, validation, and test sets
x1_train, x2_train, y_train = train_data['X1'].values, train_data['X2'].values, train_data['Label'].values
x1_val, x2_val, y_val = val_data['X1'].values, val_data['X2'].values, val_data['Label'].values
x1_test, x2_test, y_test = test_data['X1'].values, test_data['X2'].values, test_data['Label'].values

epoch_accuracies = []
epoch_f1_scores = []
epoch_precisions = []
epoch_recalls = []
epoch_auc = []
epoch_aupr = []


# 创建 DataLoader
train_loader = create_dataloader(x1_train, x2_train, y_train, batch_size=256, shuffle=True)
val_loader = create_dataloader(x1_val, x2_val, y_val, batch_size=256, shuffle=False)
test_loader = create_dataloader(x1_test, x2_test, y_test, batch_size=256, shuffle=False)
# Initialize the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
embedding_matrix_name = np.load('data_init/go_2022/go_name_embeddings_weight.npy')
embedding_tensor_name = torch.from_numpy(embedding_matrix_name).float()
model = CNN(vocab_size=47265, embedding_dim=768, pretrained_embeddings=embedding_tensor_name).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Train and evaluate the model
save_dir = "model_results"
os.makedirs(save_dir, exist_ok=True)

# 保存训练日志的文件路径
log_file = os.path.join(save_dir, "training_log.csv")
with open(log_file, "w") as f:
    f.write("epoch,train_loss,train_accuracy,train_f1,train_precision,train_recall,"
            "val_loss,val_accuracy,val_f1,val_precision,val_recall,auc,aupr\n")

# 初始化最佳指标
best_val_f1 = 0.0

# Train and evaluate the model
epochs = 10
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loss, train_accuracy, train_f1, train_precision, train_recall = train_epoch(train_loader, model, criterion, optimizer)
    val_loss, val_accuracy, val_f1, val_precision, val_recall, auc_out, aupr = evaluate_epoch(val_loader, model, criterion)

    # 记录日志
    with open(log_file, "a") as f:
        f.write(f"{epoch + 1},{train_loss},{train_accuracy},{train_f1},{train_precision},{train_recall},"
                f"{val_loss},{val_accuracy},{val_f1},{val_precision},{val_recall},{auc_out},{aupr}\n")

    # 保存最佳模型
    # if val_f1 > best_val_f1:
    #     best_val_f1 = val_f1
    #     torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    #     print("Best model saved!")

# Final evaluation on the test set
print("\nFinal evaluation on the test set")
all_preds, all_labels, probabilities = [], [], []
with torch.no_grad():
    for x1, x2, label in test_loader:
        outputs = model(x1.to(device), x2.to(device))
        _, predicted = torch.max(outputs, 1)
        all_probabilities = F.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.numpy())
        probabilities.extend(all_probabilities.cpu().numpy())

accuracy, f1, precision, recall = calculate_metrics(all_labels, all_preds)
auc_out = roc_auc_score(all_labels, probabilities)
aupr = average_precision_score(all_labels, probabilities)

# 保存测试集结果
results_file = os.path.join(save_dir, "test_results.csv")
test_data = pd.DataFrame({"true_labels": all_labels, "predicted_labels": all_preds, "probabilities": probabilities})
test_data.to_csv(results_file, index=False)
print(f"Test results saved to {results_file}")

# 打印测试结果
print(f"{accuracy}")
print(f"{f1}")
print(f"{precision}")
print(f"{recall}")
print(f"{auc_out}")
print(f"{aupr}")

# 绘制 ROC 和 AUPR 曲线
def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_out:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.savefig(save_path)
    plt.close()

def plot_aupr_curve(y_true, y_scores, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f"AUPR = {aupr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.savefig(save_path)
    plt.close()

# plot_roc_curve(all_labels, probabilities, os.path.join(save_dir, "roc_curve.png"))
# plot_aupr_curve(all_labels, probabilities, os.path.join(save_dir, "aupr_curve.png"))
# print(f"ROC and AUPR curves saved to {save_dir}")