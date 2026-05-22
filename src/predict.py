# predict.py
# =========================================
# LOAD MODEL ĐÃ TRAIN
# DỰ ĐOÁN CHURN CHO APP.PY
# =========================================

import joblib
import pandas as pd

# =========================================
# 1. LOAD MODEL
# =========================================

loaded_data = joblib.load("models/best_model.pkl")

model = loaded_data["model"]
scaler = loaded_data["scaler"]
feature_columns = loaded_data["feature_columns"]
model_name = loaded_data["model_name"]

print(f"Loaded model: {model_name}")

# =========================================
# 2. HÀM PREPROCESS INPUT
# =========================================

def preprocess_input(input_data):
    """
    input_data: dict hoặc DataFrame
    """

    # Nếu input là dict -> convert DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])

    else:
        df = input_data.copy()

    # Xóa CustomerID nếu có
    df = df.drop(columns=["CustomerID"], errors="ignore")

    # Đồng bộ cột với train
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df

# =========================================
# 3. HÀM PREDICT
# =========================================

def predict_churn(input_data):
    """
    Return:
    {
        "prediction": 0/1,
        "probability": float,
        "label": "Churn" / "No Churn"
    }
    """

    # preprocess
    df = preprocess_input(input_data)

    # scale nếu model cần scaler
    if scaler is not None:
        model_input = scaler.transform(df)
    else:
        model_input = df

    # predict
    prediction = model.predict(model_input)[0]

    # probability churn = 1
    probability = model.predict_proba(model_input)[0][1]

    # label
    label = "Churn" if prediction == 1 else "No Churn"

    return {
        "prediction": int(prediction),
        "probability": float(round(probability, 4)),
        "label": label
    }