## 1. Data Dictionary (Danh mục Dữ liệu)

Dưới đây là bảng mô tả ý nghĩa và kiểu dữ liệu của các trường thông tin trong bộ dữ liệu:

| Tên cột | Ý nghĩa (Description) | Loại (Type) |
| :--- | :--- | :--- |
| **CustomerID** | Mã định danh duy nhất của khách hàng | Numerical |
| **Churn** | Biến mục tiêu (1: Rời bỏ, 0: Ở lại) | Categorical |
| **Tenure** | Thời gian khách hàng gắn bó với công ty | Numerical |
| **PreferredLoginDevice** | Thiết bị đăng nhập ưu thích | Categorical |
| **CityTier** | Cấp bậc thành phố | Categorical |
| **WarehouseToHome** | Khoảng cách từ kho đến nhà khách hàng | Numerical |
| **PreferredPaymentMode** | Phương thức thanh toán ưu thích | Categorical |
| **Gender** | Giới tính khách hàng | Categorical |
| **HourSpendOnApp** | Số giờ dành cho ứng dụng/website | Numerical |
| **NumberOfDeviceRegistered** | Tổng số thiết bị đã đăng ký | Numerical |
| **PreferedOrderCat** | Danh mục mặt hàng ưu thích | Categorical |
| **SatisfactionScore** | Điểm số hài lòng về dịch vụ | Numerical |
| **MaritalStatus** | Tình trạng hôn nhân | Categorical |
| **NumberOfAddress** | Tổng số địa chỉ đã thêm | Numerical |
| **Complain** | Có khiếu nại trong tháng trước hay không | Categorical |
| **OrderAmountHikeFromlastYear** | Tỉ lệ tăng đơn hàng so với năm ngoái | Numerical |
| **CouponUsed** | Tổng số coupon đã dùng trong tháng trước | Numerical |
| **OrderCount** | Tổng số đơn hàng trong tháng trước | Numerical |
| **DaySinceLastOrder** | Số ngày kể từ đơn hàng cuối cùng | Numerical |
| **CashbackAmount** | Số tiền hoàn lại trung bình | Numerical |

---

## 2. Quy trình xử lý dữ liệu (Data Cleaning Flow)

Toàn bộ dữ liệu đã được đi qua chuỗi xử lý (Pipeline) sau để đảm bảo chất lượng đầu vào cho mô hình:

### **Bước 1: Chia tách dữ liệu (Data Split)**
- **Hành động:** Tách dữ liệu gốc thành hai bộ `train.csv` (80%) và `test.csv` (20%).
- **Mục đích:** Tránh hiện tượng rò rỉ dữ liệu (Data Leakage) và đánh giá mô hình khách quan.

### **Bước 2: Vá lỗ hổng (Missing Value Imputation)**
- **Hành động:** Điền giá trị trống cho các biến số bằng giá trị **Median** (Trung vị) của tập Train.
- **Các cột chính:** `Tenure`, `WarehouseToHome`, `DaySinceLastOrder`, `OrderCount`,...

### **Bước 3: Lọc nhiễu (Handling Outliers)**
- **Hành động:** Thực hiện **Capping (Clip)** các giá trị nằm ngoài biên IQR (1.5) để giới hạn ảnh hưởng của dữ liệu dị biệt.
- **Các cột chính:** `WarehouseToHome`, `Tenure`, `DaySinceLastOrder`.

### **Bước 4: Số hóa (Categorical Encoding)**
- **Hành động:** Chuyển đổi các cột chữ (`object`) sang định dạng số bằng phương pháp **One-Hot Encoding**.
- **Kỹ thuật:** Sử dụng `drop_first=True` để tối ưu hóa đặc trưng và tránh bẫy biến giả.

### **Bước 5: Xuất xưởng (Final Export)**
- **Hành động:** Lưu kết quả cuối cùng vào file **`cleaned_data.csv`**.
- **Kết quả:** Dữ liệu hoàn toàn ở dạng số, sẵn sàng cho việc tính toán tương quan và huấn luyện mô hình.