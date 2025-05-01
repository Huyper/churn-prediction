# Churn Prediction API

Dự án dự đoán khách hàng rời bỏ dịch vụ, bao gồm mô hình + encoder + data dictionary.

## Cấu trúc

- `models/churn_model.pkl`: mô hình chính
- `models/churn_encoder.pkl`: encode dữ liệu đầu vào
- `models/churn_data_dict.pkl`: định nghĩa cấu trúc input (ví dụ: thứ tự feature)

## Sử dụng

1. Cài thư viện:
    ```bash
    pip install -r requirements.txt
    ```

2. Chạy server:
    ```bash
    python app.py
    ```

3. Gửi POST request:
    ```bash
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"feature1": value, "feature2": value, ...}'
    ```

## Triển khai

Có thể deploy lên Heroku, Render hoặc Railway.

## Ghi chú

Sau khi giải nén, bạn cần thêm 2 file sau vào thư mục `models/`:
- `churn_model.pkl`
- `churn_data_dict.pkl`
