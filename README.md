# Movie Rating Prediction using Stacking Regression (From Scratch)

## Giới thiệu
Dự án này xây dựng mô hình **Stacking Regression** để dự đoán rating phim, **tự cài đặt thuật toán học máy từ đầu, không sử dụng các model có sẵn từ `sklearn`.

Mô hình bao gồm:
- **Base models**: KNN Regression, Linear Regression, Random Forest Regression  
- **Meta model**: Ridge Regression  

Mục tiêu là đánh giá khả năng cải thiện độ chính xác của stacking so với từng mô hình đơn lẻ.

---

## Giải thích chi tiết từng thư mục

###  `data/`
Chứa toàn bộ dữ liệu của dự án.

- `raw/`:  
  Dữ liệu gốc (CSV) về phim và rating, **chưa qua xử lý**.

- `processed/`:  
  Dữ liệu sau khi:
  - Làm sạch
  - Chuẩn hóa
  - Tách train / test  

---

### `src/`
Chứa **toàn bộ code logic**

#### `utils/`
Các hàm dùng chung cho toàn bộ project:
- `distance.py`: các hàm tính khoảng cách (Euclidean, Manhattan, ...)
- `metrics.py`: các metric đánh giá như MSE, RMSE, MAE

---

#### `models/`

- `knn.py`:  

- `linear_regression.py`:  

- `ridge_regression.py`:  
  - Dùng cho meta model

- `random_forest.py`:  

- `stacking.py`:  
  Cài đặt **Stacking Regression pipeline**:
  - Train base models
  - Lấy prediction của base models làm feature mới
  - Train meta model trên các prediction này
---

---

###  `notebooks/`
Notebook dùng cho:
- Trực quan dữ liệu
- Thử nghiệm
- Giải thích từng bước (phục vụ báo cáo)
- Làm sạch, chuẩn hóa
---

###  `results/`
Chứa kết quả dánh giá:
- Biểu đồ
- Bảng so sánh

---

### `main.py`
1. Load dữ liệu
2. Train base models
3. Train stacking model
4. Đánh giá và in kết quả

### Mấy file tui bỏ tạm để push code không mất folder thui nhé