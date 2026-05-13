# Decision Tree Report Notes

## 1. Mục tiêu

Mô hình Decision Tree được sử dụng làm baseline phi tuyến cho bài toán phát hiện xâm nhập mạng.
Mục tiêu là phân loại lưu lượng mạng thành hai lớp: Normal và Attack.

## 2. Cấu hình mô hình

Model: Decision Tree Classifier
Criterion: gini
Max depth: 12
Min samples split: 20
Min samples leaf: 10
Class weight: balanced

Tham số `class_weight='balanced'` được sử dụng để giảm ảnh hưởng của mất cân bằng nhãn.
Giới hạn `max_depth` giúp giảm nguy cơ cây học quá sâu và overfitting.

## 3. Kết quả đánh giá

Accuracy: 0.9167
Precision Attack: 0.9013
Recall Attack: 0.9531
F1-score Attack: 0.9265
ROC-AUC: 0.9719
PR-AUC: 0.9662

Trong bài toán IDS, Recall của lớp Attack là chỉ số quan trọng vì bỏ sót tấn công có thể gây rủi ro bảo mật.

## 4. Cấu trúc cây

Tree depth: 12
Number of leaves: 424

Decision Tree có ưu điểm là dễ giải thích thông qua các luật rẽ nhánh và feature importance.

## 5. Ma trận nhầm lẫn

True Negative: 32269
False Positive: 4731
False Negative: 2124
True Positive: 43208

False Negative là các mẫu tấn công bị dự đoán nhầm thành bình thường, cần được quan tâm trong hệ thống IDS.

## 6. Biểu đồ đã xuất

Các biểu đồ được lưu tại:

reports/figures/models/decision_tree/

Bao gồm:

- Confusion matrix
- Normalized confusion matrix
- ROC curve
- Precision-Recall curve
- Probability distribution
- Threshold analysis
- Prediction distribution donut chart
- Feature importance
- Feature importance lollipop chart
- Tree structure summary
- Decision tree preview
- Metrics summary table

## 7. Nhận xét

Decision Tree có khả năng học quan hệ phi tuyến tốt hơn Logistic Regression và dễ giải thích hơn các mô hình ensemble.
Tuy nhiên, Decision Tree đơn lẻ có thể nhạy với dữ liệu và dễ overfitting nếu không giới hạn độ sâu.
Kết quả từ mô hình này được dùng làm baseline phi tuyến trước khi thử Random Forest, XGBoost hoặc LightGBM.