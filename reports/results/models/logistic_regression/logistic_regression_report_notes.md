# Logistic Regression Report Notes

## 1. Mục tiêu

Mô hình Logistic Regression được sử dụng làm baseline tuyến tính cho bài toán phát hiện xâm nhập mạng.
Mục tiêu là phân loại lưu lượng mạng thành hai lớp: Normal và Attack.

## 2. Cấu hình mô hình

Model: Logistic Regression
Penalty: l2
C: 1.0
Solver: lbfgs
Max iterations: 2000
Class weight: balanced

Tham số `class_weight='balanced'` được sử dụng để giảm ảnh hưởng của mất cân bằng nhãn.

## 3. Kết quả đánh giá

Accuracy: 0.8356
Precision Attack: 0.8022
Recall Attack: 0.9310
F1-score Attack: 0.8618
ROC-AUC: 0.9560
PR-AUC: 0.9666

Trong bài toán IDS, Recall của lớp Attack là chỉ số quan trọng vì bỏ sót tấn công có thể gây rủi ro bảo mật.

## 4. Ma trận nhầm lẫn

True Negative: 26591
False Positive: 10409
False Negative: 3128
True Positive: 42204

False Negative là các mẫu tấn công bị dự đoán nhầm thành bình thường, cần được quan tâm khi đánh giá hệ thống IDS.

## 5. Biểu đồ đã xuất

Các biểu đồ được lưu tại:

reports/figures/logistic_regression/

Bao gồm:

- Confusion matrix
- Normalized confusion matrix
- ROC curve
- Precision-Recall curve
- Probability distribution
- Threshold analysis
- Prediction distribution donut chart
- Top coefficients
- Metrics summary table

## 6. Nhận xét

Logistic Regression là mô hình đơn giản, dễ giải thích và có tốc độ huấn luyện nhanh.
Tuy nhiên, vì đây là mô hình tuyến tính nên khả năng học các quan hệ phi tuyến trong dữ liệu mạng có thể bị hạn chế.
Kết quả từ mô hình này được dùng làm mốc so sánh với các mô hình mạnh hơn như Decision Tree, Random Forest, XGBoost hoặc LightGBM.