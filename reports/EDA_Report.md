# Báo cáo Phân tích Khám phá Dữ liệu (EDA) - Dự án Churn Prediction

## 1. Tổng quan dự án
Báo cáo này tóm tắt các kết quả chính từ giai đoạn Phân tích Khám phá Dữ liệu (EDA) và Làm sạch dữ liệu cho bài toán dự đoán khách hàng rời bỏ (Churn Prediction) trong lĩnh vực Thương mại điện tử. Mục tiêu là chuẩn bị một bộ dữ liệu "sạch" và cung cấp các hiểu biết (insights) quan trọng cho nhóm xây dựng mô hình (Modeling).

---

## 2. Thống kê Dữ liệu Tổng quát
- **Tổng số dòng dữ liệu:** 5,630 dòng.
- **Số lượng đặc trưng ban đầu:** 20 cột.
- **Số lượng đặc trưng sau xử lý (Encoding):** 31 cột.
- **Biến mục tiêu:** `Churn` (1: Rời bỏ, 0: Ở lại).

---

## 3. Xử lý Giá trị thiếu (Missing Values)
Dữ liệu ban đầu ghi nhận tình trạng thiếu hụt đồng loạt ở nhiều cột quan trọng (chiếm khoảng 4-5% tổng dữ liệu).

| Cột | Số lượng thiếu | Chiến thuật xử lý |
| :--- | :--- | :--- |
| `Tenure` | 213 | Điền bằng **Median (Trung vị)** |
| `WarehouseToHome` | 206 | Điền bằng **Median (Trung vị)** |
| `OrderCount` | 212 | Điền bằng **Median (Trung vị)** |
| `DaySinceLastOrder` | 247 | Điền bằng **Median (Trung vị)** |
| `HourSpendOnApp` | 197 | Điền bằng **Median (Trung vị)** |

**Lý do dùng Median:** Tránh ảnh hưởng bởi các giá trị ngoại lai (outliers) trong các phân phối lệch (skewed distribution). Sau khi xử lý, tập dữ liệu không còn giá trị thiếu (0 Null).

---

## 4. Xử lý Điểm dị biệt (Outliers)
Phân tích qua biểu đồ Boxplot cho thấy một số biến có giá trị ngoại lai cực đoan.

| Cột | % Outliers | Hành động | Lý do |
| :--- | :--- | :--- | :--- |
| `WarehouseToHome` | 0.02% | **Capping (IQR)** | Tỷ lệ cực thấp, tránh làm nhiễu khoảng cách. |
| `Tenure` | 0.04% | **Capping (IQR)** | Tỷ lệ cực thấp, giữ lại xu hướng chung. |
| `OrderCount` | 12.10% | **Giữ nguyên** | Tỷ lệ cao, đây là nhóm khách hàng VIP (tín hiệu quan trọng). |
| `CouponUsed` | 11.15% | **Giữ nguyên** | Đây là nhóm khách hàng "săn deal", không phải lỗi. |

---

## 5. Phân tích Biến mục tiêu (Churn)
- **Tỷ lệ khách hàng ở lại (0):** 83.17%
- **Tỷ lệ khách hàng rời bỏ (1):** 16.83%

**Đánh giá:** Đây là bài toán **Dữ liệu mất cân bằng (Imbalanced Data)**. Nhóm Modeling cần tập trung tối ưu chỉ số **Recall** và **F1-Score** thay vì Accuracy để tránh bỏ sót nhóm khách hàng rời bỏ.

---

## 6. Các nhân tố ảnh hưởng hàng đầu (Correlation Analysis)
Dựa trên phân tích tương quan (Correlation) và biểu đồ Heatmap, 3 yếu tố có tác động mạnh nhất đến việc khách hàng rời bỏ là:

1.  **Complain (0.248):** Những khách hàng từng khiếu nại có nguy cơ rời bỏ cao vượt trội. Đây là "Biến số vàng" cho mô hình.
2.  **MaritalStatus_Single (0.178):** Nhóm khách hàng độc thân có xu hướng ít ràng buộc và dễ rời bỏ hơn.
3.  **PreferedOrderCat_Mobile Phone (0.164):** Nhóm mua hàng điện tử/điện thoại thường có tính rủi ro cao về độ trung thành.

---

## 7. Kết luận và Bàn giao
- **Trạng thái dữ liệu:** Đã được làm sạch, vá lỗi và chuẩn hóa sang dạng số (One-Hot Encoding).
- **Phân chia dữ liệu:** Đã tách riêng tập **Train (80%)** và **Test (20%)** để đảm bảo khách quan.
- **Khuyến nghị cho Modeling:** Sử dụng các thuật toán mạnh như **XGBoost** hoặc **Random Forest** để xử lý tốt Outliers đã giữ lại và tình trạng mất cân bằng dữ liệu.

---
**Người báo cáo:** AIE Data Advisor
**Dự án:** AI Conquer 2026 - Churn Prediction
