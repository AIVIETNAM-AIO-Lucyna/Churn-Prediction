# E-Commerce Customer Churn Prediction

## Project Overview

This project predict whether a customer is likely to churn or stay with the service (churn) or continue using the service based on customer behavior and transaction data.
## Dataset

Dataset download:

[Google Drive Dataset](https://drive.google.com/drive/folders/1fJryIhGXF4S1N6PcA0CEZQmo1WwqGAkt?usp=sharing)

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
- Batch prediction using CSV upload
- Download prediction results as CSV

### Evaluation Metrics

Models were evaluated using:

- Accuracy
- Recall
- F1-score
- ROC-AUC

### Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Streamlit

## Deploy

Website deploy: 
[Web](https://prediction-arasaka.streamlit.app)

## Demo Video

[![Watch the demo](https://img.youtube.com/vi/vCx9BS2O-HY/maxresdefault.jpg)](https://youtu.be/vCx9BS2O-HY)

## Run App

Note: Install python --version 3.11 or 3.12

### 1. Create and activate virtual environment

**Windows**

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux**

```bash
python3.12 -m venv .venv
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

## Project Structure

```text
Churn-Prediction/
│
├── app/                                
│    └── app.py                          # Streamlit UI
│
├── data/
│     └──processed/                     
│           ├── train.csv
│           ├── test.csv
│           ├── cleaned_data.csv        # Train dataset
│           └── cleaned_test_data.csv   # Val dataset
│
├── models/                       
│    └── best_model.pkl                  # Trained model
│
├── notebooks/                    
│    ├── eda.ipynb
│    └── modeling.ipynb
│
├── reports/        
│    ├── screenshot/
│    ├── data_dictionary.md
│    ├── feature_importance.png
│    ├── model_results.csv
│    └── test_checklist.md
│
├── src/      
│    ├── eda.py                          # Data preprocessing and EDA                    
│    ├── modeling.py                     # Training and model comparison
│    └── predict.py                      # Prediction pipeline
│
├── .gitignore
├── README.md
└── requirements.txt
```