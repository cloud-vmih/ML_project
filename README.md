# Dự đoán Rating Phim sử dụng Stacking Regression (Tự cài đặt từ đầu)

## Giới thiệu
Dự án này xây dựng mô hình **Stacking Regression** để dự đoán rating phim, **tự cài đặt thuật toán học máy từ đầu, không sử dụng các model có sẵn từ [`sklearn`]**.

Mô hình bao gồm:
- **Base models**: KNN Regression, Linear Regression, Random Forest Regression  
- **Meta model**: Ridge Regression  

Mục tiêu là đánh giá khả năng cải thiện độ chính xác của stacking so với từng mô hình đơn lẻ.

## Tính năng
- Ghép nối và làm sạch dữ liệu từ các file CSV thô.
- Cài đặt tùy chỉnh các mô hình hồi quy.
- Tinh chỉnh siêu tham số cho base và meta models.
- Huấn luyện mô hình, đánh giá và lưu trữ.
- So sánh dự đoán giữa các mô hình.
- UI tương tác

## Cài đặt
1. Cài đặt các phụ thuộc:
   ```bash
   pip install pandas numpy scikit-learn, pandas, numpy, streamlit
   ```
2. Đảm bảo sử dụng Python 3.12+.

## Cách sử dụng
Chạy script chính để xử lý dữ liệu, huấn luyện mô hình và đánh giá:
```bash
python main.py
```
Chạy streamlit (nên chạy - model đã được train và lưu lại, có thể test trên UI)
```bash
streamlit run app.py
```
Điều này sẽ (nếu chạy main.py):
- Ghép nối dữ liệu thô thành [`data/raw/all_movies_data.csv`](data/raw/all_movies_data.csv ) (nếu chưa có).
- Lọc dữ liệu theo năm 1975–2025.
- Làm sạch và tiền xử lý dữ liệu.
- Huấn luyện base models (Linear, KNN, Random Forest).
- Huấn luyện stacking ensemble.
- Lưu mô hình vào [`models_trained`](models_trained ) và metrics vào file JSON.
- Tạo file CSV so sánh tại [`data/processed/model_check.csv`](data/processed/model_check.csv ).

## Cấu trúc dự án
- [`data`](data ): Các file dữ liệu
  - `raw/`: Dữ liệu gốc (CSV) về phim và rating, **chưa qua xử lý**.
  - `processed/`: Dữ liệu sau khi làm sạch, chuẩn hóa và tách train/test.
  - `split/`: Dữ liệu train/test đã tách.
- [`src`](src ): Mã nguồn
  - `preprocess/`: Ghép nối và làm sạch dữ liệu.
  - [`models/`](src/models/__init__.py ): Cài đặt tùy chỉnh mô hình.
  - [`evaluation/`](src/evaluation/__init__.py ): Tính toán metrics.
  - `utils/`: Các hàm tiện ích (khoảng cách, metrics).
- [`notebooks`](notebooks ): Notebook Jupyter cho EDA và thử nghiệm.
- [`results`](results ): Biểu đồ và so sánh kết quả đánh giá.
- [`models_trained`](models_trained ): Mô hình đã huấn luyện và metrics đã lưu.
- [`main.py`](main.py ): Script chính để chạy pipeline.

## Mô hình và Metrics
- Mô hình được lưu dưới dạng `.pkl` trong [`models_trained`](models_trained ).
- Metrics (RMSE, MAE, v.v.) được lưu dưới dạng JSON trong [`models_trained`](models_trained ).
- Ví dụ metrics:
  ```json
  {
    "RMSE": 0.85,
    "MAE": 0.65,
    "R2": 0.72
  }
  ```

## Kết quả
Mô hình stacking thường vượt trội hơn các base models riêng lẻ. Kiểm tra [`results`](results ) cho biểu đồ và [`data/processed/model_check.csv`](data/processed/model_check.csv ) cho so sánh dự đoán.
