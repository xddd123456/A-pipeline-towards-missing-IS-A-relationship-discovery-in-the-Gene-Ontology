import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "10"  # 改成你CPU的核数
# 1. 加载数据和嵌入
embedding_matrix = np.load('data_init/go_2022/go_name_embeddings_weight.npy')
data = pd.read_csv("data/go_2022/train_2.csv", sep='\t', header=None, names=["X1", "X2", "Label"])

# 2. 特征处理（拼接两个概念的预训练嵌入）
def get_pair_features(x1_ids, x2_ids):
    features = []
    for id1, id2 in zip(x1_ids, x2_ids):
        emb1 = embedding_matrix[int(id1)]
        emb2 = embedding_matrix[int(id2)]
        features.append(np.concatenate([emb1, emb2]))
    return np.array(features)

X_all = get_pair_features(data["X1"].values, data["X2"].values)
y_all = data["Label"].values

# 3. 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, random_state=12)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=12)

# 4. 定义统一评估函数
def evaluate(model, X, y):
    preds = model.predict(X)
    try:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
        else:
            scores = model.predict_proba(X)[:, 1]
    except:
        scores = preds
    acc = accuracy_score(y, preds)
    pre = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, scores)
    aupr = average_precision_score(y, scores)
    return {"ACC": acc, "PRE": pre, "RECALL": rec, "F1": f1, "AUC": auc, "AUPR": aupr}

# 5. 定义模型列表
models = {
    # "KNN": Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    # ]),
    "GaussianNB": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ]),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=20, early_stopping=True, random_state=42))
    ]),
    "XGBoost": XGBClassifier(
        n_estimators=100, use_label_encoder=False, eval_metric='logloss',
        tree_method='hist', verbosity=0, n_jobs=-1, random_state=42
    )
}

# 6. 训练和评估
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_val, y_val)
    results[name] = metrics
    print(f"{name} Validation Results: {metrics}\n")

# 7. 输出所有模型的指标汇总
print("=== Summary on Validation Set ===")
for name, metric in results.items():
    print(f"{name}: "
          f"ACC={float(metric['ACC']):.4f}, "
          f"PRE={float(metric['PRE']):.4f}, "
          f"RECALL={float(metric['RECALL']):.4f}, "
          f"F1={float(metric['F1']):.4f}, "
          f"AUC={float(metric['AUC']):.4f}, "
          f"AUPR={float(metric['AUPR']):.4f}")