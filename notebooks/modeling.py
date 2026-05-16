# =========================
# File: notebooks/modeling.ipynb
# Task: Chạy 4 experiment, tạo model_results.csv, lưu best_model.pkl
# =========================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import os

# -----------------------
# 1. Tạo thư mục lưu trữ
# -----------------------
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------
# 2. Đọc dữ liệu đã clean
# -----------------------
dataset = pd.read_csv("data/processed/cleaned_data.csv")

# Tách X và y
X = dataset.drop(["CustomerID", "Churn"], axis=1, errors="ignore")
y = dataset["Churn"]

# Chia train/test (20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# 3. Định nghĩa các experiment
# -----------------------
experiments = []

# EXP01: No Upsampling, No Scale, XGBoost
exp1_model = xgb.XGBClassifier(random_state=42, n_estimators=200, eval_metric="logloss")
exp1_model.fit(X_train, y_train)
y_pred = exp1_model.predict(X_test)
y_proba = exp1_model.predict_proba(X_test)[:, 1]

experiments.append({
    "Model": "XGBoost",
    "Upsampling": "No",
    "Scale": "No",
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba),
    "Model_obj": exp1_model
})

# EXP02: No Upsampling, Scale, Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

exp2_model = LogisticRegression(random_state=42, max_iter=1000)
exp2_model.fit(X_train_scaled, y_train)
y_pred = exp2_model.predict(X_test_scaled)
y_proba = exp2_model.predict_proba(X_test_scaled)[:, 1]

experiments.append({
    "Model": "Logistic Regression",
    "Upsampling": "No",
    "Scale": "Yes",
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba),
    "Model_obj": exp2_model
})

# EXP03: Upsampling SMOTE, No Scale, Gradient Boosting
sm = SMOTE(k_neighbors=5, random_state=42)
X_train_resample, y_train_resample = sm.fit_resample(X_train, y_train)

exp3_model = GradientBoostingClassifier(random_state=42)
exp3_model.fit(X_train_resample, y_train_resample)
y_pred = exp3_model.predict(X_test)
y_proba = exp3_model.predict_proba(X_test)[:, 1]

experiments.append({
    "Model": "Gradient Boosting",
    "Upsampling": "Yes",
    "Scale": "No",
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba),
    "Model_obj": exp3_model
})

# EXP04: Upsampling SMOTE, Scale, Random Forest
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_resample, y_train_resample = sm.fit_resample(X_train_scaled, y_train)

exp4_model = RandomForestClassifier(random_state=42, n_estimators=200)
exp4_model.fit(X_train_resample, y_train_resample)
y_pred = exp4_model.predict(X_test_scaled)
y_proba = exp4_model.predict_proba(X_test_scaled)[:, 1]

experiments.append({
    "Model": "Random Forest",
    "Upsampling": "Yes",
    "Scale": "Yes",
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba),
    "Model_obj": exp4_model
})

# Tạo thư mục reports nếu chưa tồn tại
os.makedirs("reports", exist_ok=True)

# Tìm model tốt nhất theo F1-score
best_exp = max(experiments, key=lambda x: x["F1"])

# Kiểm tra model có attribute feature_importances_ hay không
if hasattr(best_exp["Model_obj"], "feature_importances_"):
    feature_names = X.columns
    importances = best_exp["Model_obj"].feature_importances_

    # Tạo DataFrame sắp xếp theo importance
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    feat_imp = feat_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)

    # In bảng top 10 feature ra console
    print("Top 10 features ảnh hưởng đến Churn:")
    print(feat_imp.head(10))

    # Vẽ top 20 features
    plt.figure(figsize=(10,6))
    plt.barh(feat_imp['feature'][:20][::-1], feat_imp['importance'][:20][::-1], color='skyblue')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 20 Feature Importance - {best_exp['Model']}")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=300)
    plt.show()
else:
    print(f"Model {best_exp['Model']} không có attribute feature_importances_, không thể vẽ feature importance.")

# -----------------------
# 4. Tạo DataFrame CSV
# -----------------------
results_df = pd.DataFrame([{
    "Model": e["Model"],
    "Upsampling": e["Upsampling"],
    "Scale": e["Scale"],
    "Accuracy": e["Accuracy"],
    "F1": e["F1"],
    "Recall": e["Recall"],
    "AUC": e["AUC"]
} for e in experiments])

# Lưu CSV
results_df.to_csv("reports/model_results.csv", index=False)
print("Saved CSV: reports/model_results.csv")

# -----------------------
# 5. In kết quả và nhận xét
# -----------------------
print("\n=== Model Results ===")
print(results_df)

print("\nBest model based on F1-score:", best_exp["Model"])
print("Accuracy: {:.3f}, F1: {:.3f}, Recall: {:.3f}, AUC: {:.3f}".format(
    best_exp["Accuracy"], best_exp["F1"], best_exp["Recall"], best_exp["AUC"]
))

# Nhận xét: so sánh các chỉ số
print("\n=== Model Comparison ===")
for e in experiments:
    print(f"{e['Model']}: Accuracy={e['Accuracy']:.3f}, F1={e['F1']:.3f}, Recall={e['Recall']:.3f}, AUC={e['AUC']:.3f}")

# -----------------------
# 6. Lưu model tốt nhất
# -----------------------
joblib.dump(best_exp["Model_obj"], "models/best_model.pkl")
print("Saved best model to: models/best_model.pkl")

data_1 = joblib.load("models/best_model.pkl")
print(data_1)