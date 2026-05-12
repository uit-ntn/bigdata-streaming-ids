# EDA Report Notes

## Nội dung đã phân tích

1. Tổng quan số dòng, số cột của train/test.
2. Kiểm tra kiểu dữ liệu và số lượng giá trị duy nhất.
3. Kiểm tra missing values.
4. Kiểm tra duplicate rows.
5. Phân phối nhãn Normal/Attack.
6. Phân phối các loại tấn công trong `attack_cat`.
7. Phân tích các thuộc tính phân loại như `proto`, `service`, `state`.
8. Phân tích các thuộc tính số.
9. So sánh giá trị trung bình của các đặc trưng giữa Normal và Attack.
10. Phân tích tương quan giữa các đặc trưng số và nhãn `label`.
11. So sánh khác biệt cơ bản giữa train và test.

## Nhận xét dùng cho báo cáo

- Dataset có thể dùng cho bài toán phân loại nhị phân với `label`: 0 là Normal, 1 là Attack.
- Cột `attack_cat` có thể dùng cho phân tích phân phối loại tấn công hoặc mở rộng sang bài toán phân loại đa lớp.
- Các cột dạng chuỗi như `proto`, `service`, `state` cần được mã hóa trước khi đưa vào mô hình học máy.
- Các cột số có thang đo khác nhau nên cần cân nhắc chuẩn hóa khi dùng các mô hình như Logistic Regression.
- Với các mô hình cây như Decision Tree, Random Forest, XGBoost hoặc LightGBM, việc chuẩn hóa không bắt buộc nhưng vẫn cần xử lý dữ liệu phân loại.
- Nếu phân phối nhãn mất cân bằng, cần ưu tiên Recall và F1-score của lớp Attack thay vì chỉ nhìn Accuracy.
