# Checklisst

## Data
- Data Quality
    - Không còn missing values (hoặc đã xử lý rõ ràng)
    - Không duplicate (dữ liệu giống nhau trong bảng dữ liệu)
    - Outliers được xử lý hoặc giải thích (các ngoại lệ, như quá cao hoặc quá thấp)
- EDA
    - Có churn rate
    - Có distribution (vẽ Histogram / KDE)
    - Có correlation
    - Có phân tích theo feature (gender, tenure…)
- Output
    - Có file final_data.csv
    - có train.csv, test.csv
- góp ý:
    - nên bỏ cột ID trước khi traning

## Model
-Modeling Process
    - Có baseline model
    - Có 4 model để so sánh
-Metrics
    - Có Accuracy, Recall, F1, AUC
-Output
    - best_model.pkl
    - model_results.csv
    - feature_importance.png