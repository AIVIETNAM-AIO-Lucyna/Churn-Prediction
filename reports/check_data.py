import pandas as pd
import numpy as np

# ===== CONFIG =====
FILE_PATH = "train.csv"   # sửa đường dẫn nếu cần
TARGET = "Churn"
ID_COL = "CustomerID"

# ===== LOAD DATA =====
df = pd.read_csv(r"D:\AIO2026\project\Churn_predict\code\Churn-Prediction\data\processed\train_final.csv")

print("="*50)
print("📊 BASIC INFO")
print("="*50)
print("Shape:", df.shape)
print(df.info())

# ===== 1. DATA QUALITY =====
print("\n" + "="*50)
print("🧪 DATA QUALITY CHECK")
print("="*50)

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]

print("\n🔎 Missing values:")
print(missing if len(missing) > 0 else "No missing values ✅")

# Duplicate
dup = df.duplicated().sum()
print(f"\n🔁 Duplicate rows: {dup}")

# Data types
print("\n📌 Data types:")
print(df.dtypes)

# Basic stats
print("\n📈 Describe:")
print(df.describe())

# ===== 2. DATA INTEGRITY =====
print("\n" + "="*50)
print("🔍 DATA INTEGRITY CHECK")
print("="*50)

# Check target values
if TARGET in df.columns:
    unique_target = df[TARGET].unique()
    print(f"\n🎯 Target unique values: {unique_target}")
else:
    print("❌ Target column not found!")

# Check negative values in numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
neg_check = (df[num_cols] < 0).sum()

print("\n⚠️ Negative values check:")
print(neg_check[neg_check > 0] if neg_check.sum() > 0 else "No negative values ✅")

# ===== 3. CHURN RATE =====
print("\n" + "="*50)
print("📊 CHURN RATE")
print("="*50)

if TARGET in df.columns:
    churn_rate = df[TARGET].value_counts(normalize=True)
    print(churn_rate)
else:
    print("❌ Cannot compute churn rate")

# ===== 4. CORRELATION =====
print("\n" + "="*50)
print("🔗 CORRELATION")
print("="*50)

if TARGET in df.columns:
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    print("\nTop correlation with target:")
    print(corr[TARGET].sort_values(ascending=False))
else:
    print("❌ Cannot compute correlation")

# ===== 5. FEATURE ANALYSIS =====
print("\n" + "="*50)
print("🔍 FEATURE ANALYSIS")
print("="*50)

if TARGET in df.columns:
    # Numeric features
    print("\n📊 Numeric feature vs Churn (mean):")
    for col in num_cols:
        if col != TARGET:
            mean_vals = df.groupby(TARGET)[col].mean()
            print(f"\n{col}:")
            print(mean_vals)

    # Categorical features
    cat_cols = df.select_dtypes(include=['object']).columns

    print("\n📊 Categorical feature vs Churn (rate):")
    for col in cat_cols:
        print(f"\n{col}:")
        rate = df.groupby(col)[TARGET].mean()
        print(rate)
else:
    print("❌ Cannot perform feature analysis")

# ===== 6. WARNING SYSTEM =====
print("\n" + "="*50)
print("🚨 QA WARNINGS")
print("="*50)

warnings = []

if missing.sum() > 0:
    warnings.append("Missing values detected")

if dup > 0:
    warnings.append("Duplicate rows detected")

if ID_COL in df.columns:
    warnings.append("ID column should be dropped before training")

if TARGET in df.columns:
    imbalance = df[TARGET].value_counts(normalize=True).min()
    if imbalance < 0.2:
        warnings.append("Data is imbalanced")

if len(warnings) == 0:
    print("✅ No major issues found")
else:
    for w in warnings:
        print("⚠️", w)

print("\n" + "="*50)
print("✅ QA CHECK COMPLETED")
print("="*50)