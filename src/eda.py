import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os

# 1. Đường dẫn (Sử dụng r"" để tránh lỗi gạch chéo)
path = r"data/raw/ecommerce_churn.xlsx"

# 2. Đọc Sheet thứ 2 (sheet_name=1)
# Nếu máy báo lỗi "ImportError: Missing optional dependency 'openpyxl'", 
# bạn hãy vào Terminal gõ: pip install openpyxl
df = pd.read_excel(path, sheet_name="E Comm")

# 3. Làm sạch tên cột (đề phòng khoảng trắng thừa)
df.columns = df.columns.str.strip()

# Kiểm tra thử xem đã thấy cột Churn chưa
print("Danh sách cột đã tìm thấy:", df.columns.tolist())
print(f"Tổng số dòng dữ liệu: {len(df)}")

# 4. Nếu đã thấy cột 'Churn', tiến hành chia dữ liệu
if 'Churn' in df.columns:
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['Churn']
    )
    
    # 5. Lưu vào thư mục processed
    os.makedirs('../data/processed', exist_ok=True)
    train_df.to_csv('../data/processed/train.csv', index=False)
    test_df.to_csv('../data/processed/test.csv', index=False)
    print("--- THÀNH CÔNG: Đã chia Train/Test và lưu file! ---")
else:
    print("Vẫn chưa tìm thấy cột 'Churn'. Hãy kiểm tra lại tên Sheet!")

# Đọc lại tập train đã lưu
train_df = pd.read_csv('../data/processed/train.csv')

# Kiểm tra các cột có giá trị thiếu
missing_data = train_df.isnull().sum()
print("Các cột bị thiếu dữ liệu:\n", missing_data[missing_data > 0])

# 1. Danh sách các cột cần vá (Numerical columns)
cols_to_fix = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
    'OrderAmountHikeFromlastYear', 'CouponUsed', 
    'OrderCount', 'DaySinceLastOrder'
]

# 2. Vá lỗi bằng Median (Trung vị)
for col in cols_to_fix:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)

# 3. Kiểm tra lại xem còn lỗ hổng nào không
print("Số lượng giá trị thiếu sau khi vá:")
print(train_df.isnull().sum().sum())

# Vẽ biểu đồ đếm số lượng
sns.countplot(x='Churn', data=train_df)
plt.title('Tỷ lệ khách hàng Ở lại (0) và Rời bỏ (1)')
plt.show()

# Xem tỷ lệ phần trăm cụ thể
print(train_df['Churn'].value_counts(normalize=True) * 100)

duplicate_count = train_df.duplicated().sum()
print(f"Số lượng dòng trùng lặp: {duplicate_count}")

# Nếu có, chúng ta sẽ xóa bỏ và chỉ giữ lại dòng đầu tiên
if duplicate_count > 0:
    train_df = train_df.drop_duplicates()
    print("Đã xóa các dòng trùng lặp!")

# 1. Tìm tất cả các cột số
num_features = train_df.select_dtypes(include=['int64', 'float64']).columns

# 2. Tính toán tỷ lệ Outlier ở mỗi cột
def count_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((df[col] < lower) | (df[col] > upper)).sum()

for col in num_features:
    outlier_count = count_outliers(train_df, col)
    if outlier_count > 0:
        percentage = (outlier_count / len(train_df)) * 100
        print(f"Cột {col}: có {outlier_count} outliers ({percentage:.2f}%)")

# 1. Danh sách các cột bạn muốn vẽ (dựa trên danh sách bạn đã liệt kê)
num_features = [
    'Churn', 'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
    'NumberOfDeviceRegistered', 'NumberOfAddress', 'OrderAmountHikeFromlastYear', 
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
]

# 2. Tính toán số hàng và số cột cho khung hình (Grid)
# Ví dụ: 11 cột thì chia thành 4 hàng, mỗi hàng 3 biểu đồ
n_cols = 3
n_rows = math.ceil(len(num_features) / n_cols)

# 3. Khởi tạo khung hình
plt.figure(figsize=(15, n_rows * 4))
plt.suptitle('Biểu đồ Boxplot tổng thể - Kiểm tra Outliers', fontsize=16, y=1.02)

for i, col in enumerate(num_features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x=train_df[col], color='skyblue')
    plt.title(f'Outliers của {col}')
    plt.xlabel('') # Ẩn tên cột ở trục X cho đỡ rối

# 4. Tự động căn chỉnh để các biểu đồ không đè lên nhau
plt.tight_layout()
plt.show()

# A. ĐỊNH NGHĨA LẠI HÀM (Để máy nhớ lại cách xử lý Outlier)
def handle_outliers(df, column):
    # Sử dụng phương pháp IQR (Interquartile Range)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Capping: Thay thế giá trị ngoài biên bằng giá trị biên
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# B. THỰC HIỆN XỬ LÝ CHIẾN THUẬT [cite: 642]
# Chỉ cắt những cột có Outlier cực thấp (<1%) để tránh mất tín hiệu quan trọng 
cols_to_cap = ['WarehouseToHome', 'Tenure', 'DaySinceLastOrder']

for col in cols_to_cap:
    train_df = handle_outliers(train_df, col)

# C. CHUẨN HÓA LẦN CUỐI (Encoding) [cite: 114, 503, 642]
cat_cols = train_df.select_dtypes(include=['object']).columns
train_final = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)

# D. LƯU FILE
import os
os.makedirs('../data/processed', exist_ok=True)
train_final.to_csv('../data/processed/cleaned_data.csv', index=False)

print("--- CHÚC MỪNG: DỮ LIỆU ĐÃ SẠCH VÀ CHUẨN CHIẾN THUẬT! ---")
print(f"Số lượng cột cuối cùng: {train_final.shape[1]}")

# 1. Tính ma trận tương quan
# Lưu ý: ma trận này chỉ tính được trên dữ liệu số (đã encode) 
corr_matrix = train_final.corr()

# 2. Xem độ tương quan của các biến đối với biến mục tiêu Churn
# Sắp xếp giảm dần để thấy biến nào ảnh hưởng mạnh nhất
target_corr = corr_matrix['Churn'].sort_values(ascending=False)
print("Độ tương quan với cột Churn:")
print(target_corr)

# 3. Vẽ biểu đồ Heatmap cho toàn bộ các đặc trưng
# Chọn top các biến có tương quan cao nhất để biểu đồ không bị quá dày đặc
top_features = target_corr.index[:15] # Lấy top 15 biến tương quan mạnh nhất

plt.figure(figsize=(12, 10))
sns.heatmap(train_final[top_features].corr(), 
            annot=True, 
            fmt=".2f", 
            cmap='RdBu', # Màu đỏ (tương quan dương), Xanh (tương quan âm)
            linewidths=0.5)

plt.title('Ma trận tương quan - Top 15 đặc trưng ảnh hưởng đến Churn')
plt.show()