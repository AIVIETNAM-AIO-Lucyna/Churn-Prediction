# %%
# =========================================
# TRAIN BẰNG clean_data
# TEST BẰNG clean_test_data CÓ CHURN
# CHẠY 4 MODEL:
# - Logistic Regression
# - Random Forest
# - Gradient Boosting
# - XGBoost
# =========================================

import os
import joblib
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

warnings.filterwarnings("ignore")

print("=== BẮT ĐẦU TRAIN VÀ TEST MODEL DỰ ĐOÁN CHURN ===")

# =========================================
# 1. TẠO THƯ MỤC
# =========================================

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =========================================
# 2. LOAD DATA
# =========================================

train_path = "data/processed/cleaned_data.csv"
test_path = "data/processed/cleaned_test_data.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Kích thước train:", train_df.shape)
print("Kích thước test:", test_df.shape)

# =========================================
# 3. KIỂM TRA CỘT CHURN
# =========================================

if "Churn" not in train_df.columns:
    raise ValueError("File cleaned_data.csv không có cột Churn")

if "Churn" not in test_df.columns:
    raise ValueError("File cleaned_test_data.csv không có cột Churn")

# =========================================
# 4. TÁCH X, y
# =========================================

X_train = train_df.drop(["Churn", "CustomerID"], axis=1, errors="ignore")
y_train = train_df["Churn"]

X_test = test_df.drop(["Churn", "CustomerID"], axis=1, errors="ignore")
y_test = test_df["Churn"]

# Lưu CustomerID
if "CustomerID" in test_df.columns:
    customer_id = test_df["CustomerID"]
else:
    customer_id = pd.Series(
        range(1, len(test_df) + 1),
        name="CustomerID"
    )

# =========================================
# 5. ĐỒNG BỘ CỘT TRAIN / TEST
# =========================================

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Số cột X_train:", X_train.shape[1])
print("Số cột X_test:", X_test.shape[1])

if list(X_train.columns) == list(X_test.columns):
    print("Train/Test columns đã khớp")
else:
    print("Train/Test columns chưa khớp")

# =========================================
# 6. XỬ LÝ IMBALANCE
# =========================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

if positive_count == 0:
    scale_pos_weight = 1
else:
    scale_pos_weight = negative_count / positive_count

print("\n=== THỐNG KÊ CHURN ===")
print("Churn = 0:", negative_count)
print("Churn = 1:", positive_count)
print("scale_pos_weight:", round(scale_pos_weight, 2))

# =========================================
# 7. CẤU HÌNH SMOTE AN TOÀN
# =========================================

minority_count = y_train.value_counts().min()
k_neighbors = min(5, minority_count - 1)

smote = SMOTE(
    k_neighbors=k_neighbors,
    random_state=42
)

print("SMOTE k_neighbors:", k_neighbors)

# =========================================
# 8. LIST LƯU EXPERIMENTS
# =========================================

experiments = []

# =========================================
# EXP01 - XGBOOST
# No Scale / No SMOTE
# =========================================

print("\n=== TRAIN XGBOOST ===")

xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

experiments.append({
    "Model": "XGBoost",
    "Upsampling": "No",
    "Scale": "No",
    "Scaler": None,
    "Accuracy": accuracy_score(y_test, xgb_pred),
    "Precision": precision_score(y_test, xgb_pred),
    "Recall": recall_score(y_test, xgb_pred),
    "F1": f1_score(y_test, xgb_pred),
    "AUC": roc_auc_score(y_test, xgb_proba),
    "Model_obj": xgb_model,
    "y_pred": xgb_pred,
    "y_proba": xgb_proba
})

# =========================================
# EXP02 - LOGISTIC REGRESSION
# Scale / No SMOTE
# =========================================

print("\n=== TRAIN LOGISTIC REGRESSION ===")

lr_scaler = StandardScaler()

X_train_lr = lr_scaler.fit_transform(X_train)
X_test_lr = lr_scaler.transform(X_test)

lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

lr_model.fit(X_train_lr, y_train)

lr_pred = lr_model.predict(X_test_lr)
lr_proba = lr_model.predict_proba(X_test_lr)[:, 1]

experiments.append({
    "Model": "Logistic Regression",
    "Upsampling": "No",
    "Scale": "Yes",
    "Scaler": lr_scaler,
    "Accuracy": accuracy_score(y_test, lr_pred),
    "Precision": precision_score(y_test, lr_pred),
    "Recall": recall_score(y_test, lr_pred),
    "F1": f1_score(y_test, lr_pred),
    "AUC": roc_auc_score(y_test, lr_proba),
    "Model_obj": lr_model,
    "y_pred": lr_pred,
    "y_proba": lr_proba
})

# =========================================
# EXP03 - GRADIENT BOOSTING
# No Scale / SMOTE
# =========================================

print("\n=== TRAIN GRADIENT BOOSTING ===")

X_train_gb, y_train_gb = smote.fit_resample(
    X_train,
    y_train
)

gb_model = GradientBoostingClassifier(
    random_state=42
)

gb_model.fit(X_train_gb, y_train_gb)

gb_pred = gb_model.predict(X_test)
gb_proba = gb_model.predict_proba(X_test)[:, 1]

experiments.append({
    "Model": "Gradient Boosting",
    "Upsampling": "Yes",
    "Scale": "No",
    "Scaler": None,
    "Accuracy": accuracy_score(y_test, gb_pred),
    "Precision": precision_score(y_test, gb_pred),
    "Recall": recall_score(y_test, gb_pred),
    "F1": f1_score(y_test, gb_pred),
    "AUC": roc_auc_score(y_test, gb_proba),
    "Model_obj": gb_model,
    "y_pred": gb_pred,
    "y_proba": gb_proba
})

# =========================================
# EXP04 - RANDOM FOREST
# No Scale / SMOTE
# =========================================

print("\n=== TRAIN RANDOM FOREST ===")

X_train_rf, y_train_rf = smote.fit_resample(
    X_train,
    y_train
)

rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=200
)

rf_model.fit(X_train_rf, y_train_rf)

rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

experiments.append({
    "Model": "Random Forest",
    "Upsampling": "Yes",
    "Scale": "No",
    "Scaler": None,
    "Accuracy": accuracy_score(y_test, rf_pred),
    "Precision": precision_score(y_test, rf_pred),
    "Recall": recall_score(y_test, rf_pred),
    "F1": f1_score(y_test, rf_pred),
    "AUC": roc_auc_score(y_test, rf_proba),
    "Model_obj": rf_model,
    "y_pred": rf_pred,
    "y_proba": rf_proba
})

# =========================================
# 9. TẠO BẢNG KẾT QUẢ
# =========================================

results_df = pd.DataFrame([
    {
        "Model": e["Model"],
        "Upsampling": e["Upsampling"],
        "Scale": e["Scale"],
        "Accuracy": round(e["Accuracy"], 4),
        "Precision": round(e["Precision"], 4),
        "Recall": round(e["Recall"], 4),
        "F1": round(e["F1"], 4),
        "AUC": round(e["AUC"], 4)
    }
    for e in experiments
])

# =========================================
# 10. LƯU CSV
# =========================================

results_path = "reports/model_results.csv"

results_df.to_csv(results_path, index=False)

print("\nSaved:", results_path)

# =========================================
# 11. HIỂN THỊ KẾT QUẢ
# =========================================

print("\n=== MODEL RESULTS ===")
print(results_df)

# =========================================
# 12. CHỌN MODEL TỐT NHẤT
# =========================================

best_exp = max(experiments, key=lambda x: x["F1"])

print("\n=== BEST MODEL ===")
print("Model:", best_exp["Model"])
print("Accuracy:", round(best_exp["Accuracy"], 4))
print("Precision:", round(best_exp["Precision"], 4))
print("Recall:", round(best_exp["Recall"], 4))
print("F1:", round(best_exp["F1"], 4))
print("AUC:", round(best_exp["AUC"], 4))

# =========================================
# 13. CLASSIFICATION REPORT
# =========================================

print("\n=== CLASSIFICATION REPORT ===")

print(classification_report(
    y_test,
    best_exp["y_pred"]
))

# =========================================
# 14. CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y_test, best_exp["y_pred"])

print("\n=== CONFUSION MATRIX ===")
print(cm)

# =========================================
# 17. LƯU MODEL + SCALER
# =========================================

save_object = {
    "model": best_exp["Model_obj"],
    "scaler": best_exp["Scaler"],
    "model_name": best_exp["Model"],
    "feature_columns": X_train.columns.tolist()
}
joblib.dump(
    save_object,
    "models/best_model.pkl"
)

print("\nSaved model:")
print("models/best_model.pkl")

# =========================================
# 18. LOAD MODEL TEST
# =========================================

loaded_data = joblib.load(
    "models/best_model.pkl"
)

loaded_model = loaded_data["model"]
loaded_scaler = loaded_data["scaler"]

print("\nLoaded model:")
print(loaded_data["model_name"])

# =========================================
# 19. DEBUG PREDICT SAMPLE
# =========================================

sample = X_test.iloc[[0]]

if loaded_scaler is not None:
    sample_input = loaded_scaler.transform(sample)
else:
    sample_input = sample

sample_pred = loaded_model.predict(sample_input)
sample_proba = loaded_model.predict_proba(sample_input)

print("\n=== SAMPLE PREDICTION ===")
print("Prediction:", sample_pred)
print("Probability:", sample_proba)