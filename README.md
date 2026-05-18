# Churn-Prediction

## Project Overview

This project predicts whether an e-commerce customer is likely to leave (churn) or continue using the service based on customer behavior and transaction data.
[Dataset](https://drive.google.com/drive/folders/1fJryIhGXF4S1N6PcA0CEZQmo1WwqGAkt?usp=sharing)

The project includes an end-to-end machine learning pipeline:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and comparison
- Best model selection
- Model persistence using `.pkl`
- Real-time prediction through a Streamlit web application


### Features

- Analyze customer behavior patterns
- Predict customer churn probability
- Compare multiple machine learning models:
  - XGBoost
  - Logistic Regression
  - Gradient Boosting
  - Random Forest
- Save and load the best-performing model
- Interactive web interface with Streamlit

### Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Streamlit

---

## Run App

### 1. Create and activate virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install required libraries

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit application

```bash
streamlit run app/app.py
```

---

## Project Structure

```text
Churn-Prediction/
│
├── app/
│   └── app.py                 # Streamlit UI
│
├── src/
│   └── predict.py             # Prediction pipeline
│
├── models/
│   └── best_model.pkl         # Trained model
│
├── data/
│   └── processed/
│       └── cleaned_data.csv   # Processed dataset
│
├── notebooks/
│   ├── eda.py                 # Data preprocessing and EDA
│   └── modeling.py            # Training and model comparison
│
├── reports/
│   ├── model_results.csv
│   └── feature_importance.png
│
├── requirements.txt
└── README.md
```