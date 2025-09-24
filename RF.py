import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

# Step 1: 加载嵌入
embedding_matrix = np.load('data_init/go_2022/go_name_embeddings_weight.npy')  # shape: [47265, 768]

# Step 2: 加载配对数据
data = pd.read_csv("data/go_2022/train_2.csv", sep='\t', header=None, names=["X1", "X2", "Label"])

# Step 3: 构造特征向量
def get_pair_features(x1_ids, x2_ids, method='concat'):
    features = []
    for id1, id2 in zip(x1_ids, x2_ids):
        emb1 = embedding_matrix[int(id1)]
        emb2 = embedding_matrix[int(id2)]
        if method == 'concat':
            vec = np.concatenate([emb1, emb2])
        elif method == 'diff':
            vec = emb2 - emb1
        elif method == 'concat_diff':
            vec = np.concatenate([emb1, emb2, emb2 - emb1])
        else:
            raise ValueError("Unsupported feature construction method.")
        features.append(vec)
    return np.array(features)

X_all = get_pair_features(data["X1"].values, data["X2"].values, method='concat')
y_all = data["Label"].values

# Step 4: 数据划分
X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 5: 训练随机森林
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,
    max_features='sqrt',
    random_state=2)
rf_model.fit(X_train, y_train)

# ========== 统一评估函数 ==========
def evaluate(model, X, y, set_name=""):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    pre = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, probs)
    aupr = average_precision_score(y, probs)
    print(f"\n=== {set_name} ===")
    print(f"{acc:.4f}")
    print(f"{pre:.4f}")
    print(f"{rec:.4f}")
    print(f"{f1:.4f}")
    print(f"{auc:.4f}")
    print(f"{aupr:.4f}")
    return acc, pre, rec, f1, auc, aupr

# Step 6: 验证集评估
evaluate(rf_model, X_val, y_val, set_name="Validation Set")

# Step 7: 测试集评估
evaluate(rf_model, X_test, y_test, set_name="Test Set")
