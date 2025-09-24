import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# Step 1: 加载嵌入 + 数据
embedding_matrix = np.load('data_init/go_2022/go_name_embeddings_weight.npy')
data = pd.read_csv("data/go_2022/train_2.csv", sep='\t', header=None, names=["X1", "X2", "Label"])

# Step 2: 构造特征向量（你可以选用 'diff' 或 'concat'）
def get_pair_features(x1_ids, x2_ids, method='concat'):
    features = []
    for id1, id2 in zip(x1_ids, x2_ids):
        emb1 = embedding_matrix[int(id1)]
        emb2 = embedding_matrix[int(id2)]
        if method == 'concat':
            vec = np.concatenate([emb1, emb2])
        elif method == 'diff':
            vec = emb2 - emb1
        else:
            raise ValueError("Unsupported method.")
        features.append(vec)
    return np.array(features)

X_all = get_pair_features(data["X1"].values, data["X2"].values, method='concat')
y_all = data["Label"].values

# Step 3: 划分训练、验证、测试集
X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: 构建 SVM 模型（基于 SGD）
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SGDClassifier(
        loss='log_loss',               # SVM 损失 hinge  LR损失 log_loss
        penalty='l2',               # L2 正则
        max_iter=1000,              # 最多迭代轮数
        tol=1e-3,                   # 收敛容差
        class_weight='balanced',   # 自动平衡类别
        n_jobs=-1,                  # 多核计算
        random_state=32
    ))
])

# Step 5: 训练
svm_model.fit(X_train, y_train)

# Step 6: 评估函数
def evaluate(model, X, y, set_name=""):
    preds = model.predict(X)
    scores = model.decision_function(X)  # 连续输出，用于 AUC 和 AUPR
    acc = accuracy_score(y, preds)
    pre = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, scores)
    aupr = average_precision_score(y, scores)

    print(f"\n=== {set_name} ===")
    print(f"{acc:.4f}")
    print(f"{pre:.4f}")
    print(f"{rec:.4f}")
    print(f"{f1:.4f}")
    print(f"{auc:.4f}")
    print(f"{aupr:.4f}")
    return acc, pre, rec, f1, auc, aupr

# Step 7: 输出验证集和测试集指标
evaluate(svm_model, X_val, y_val, "Validation Set")
evaluate(svm_model, X_test, y_test, "Test Set")
