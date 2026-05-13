# Model Comparison Report Notes

## 1. Mục tiêu so sánh

Sau khi huấn luyện ba mô hình, kết quả được tổng hợp để so sánh khả năng phát hiện xâm nhập mạng.
Ba mô hình được so sánh gồm Logistic Regression, Decision Tree và Deep MLP.

## 2. Các chỉ số đánh giá

Các chỉ số được sử dụng gồm:

- Accuracy
- Precision cho lớp Attack
- Recall cho lớp Attack
- F1-score cho lớp Attack
- ROC-AUC
- PR-AUC
- Thời gian huấn luyện
- Thời gian dự đoán
- Confusion matrix components: TN, FP, FN, TP

Trong bài toán IDS, Recall và F1-score của lớp Attack được ưu tiên hơn Accuracy vì bỏ sót tấn công có thể gây rủi ro bảo mật.

## 3. Công thức điểm tổng hợp

Điểm tổng hợp được tính theo công thức:

Weighted Score = 0.35 * Recall_Attack + 0.30 * F1_Attack + 0.15 * Precision_Attack + 0.10 * ROC_AUC + 0.10 * PR_AUC

Công thức này ưu tiên Recall và F1-score của lớp Attack để phù hợp với bài toán phát hiện xâm nhập mạng.

## 4. Kết quả tốt nhất

Mô hình có Weighted Score cao nhất: Decision Tree (0.9406)
Mô hình có Recall Attack cao nhất: Decision Tree (0.9531)
Mô hình có F1 Attack cao nhất: Decision Tree (0.9265)
Mô hình có thời gian dự đoán thấp nhất: Logistic Regression (0.010121 giây)

## 5. Bảng xếp hạng mô hình

### 1. Decision Tree

- Accuracy: 0.9167
- Precision Attack: 0.9013
- Recall Attack: 0.9531
- F1 Attack: 0.9265
- ROC-AUC: 0.9719
- PR-AUC: 0.9662
- Weighted Score: 0.9406

### 2. Deep MLP

- Accuracy: 0.9063
- Precision Attack: 0.8936
- Recall Attack: 0.9420
- F1 Attack: 0.9172
- ROC-AUC: 0.9794
- PR-AUC: 0.9848
- Weighted Score: 0.9353

### 3. Logistic Regression

- Accuracy: 0.8356
- Precision Attack: 0.8022
- Recall Attack: 0.9310
- F1 Attack: 0.8618
- ROC-AUC: 0.9560
- PR-AUC: 0.9666
- Weighted Score: 0.8970

## 6. Biểu đồ đã xuất

Các biểu đồ so sánh được lưu tại:

reports/figures/model_comparison/

Bao gồm:

- Grouped metric comparison
- Attack priority metrics
- Weighted score lollipop chart
- ROC-AUC và PR-AUC comparison
- Training time comparison
- Prediction time comparison
- Confusion matrix component comparison
- False negative comparison
- False positive comparison
- Radar chart
- Combined ROC curves
- Combined Precision-Recall curves
- Summary table image

## 7. Nhận xét gợi ý

Logistic Regression là baseline tuyến tính, thường có tốc độ huấn luyện và dự đoán nhanh, nhưng có thể hạn chế khi dữ liệu có quan hệ phi tuyến.
Decision Tree dễ giải thích hơn nhờ feature importance và luật rẽ nhánh, nhưng có nguy cơ overfitting nếu cây quá sâu.
Deep MLP có khả năng học quan hệ phi tuyến phức tạp hơn, nhưng thời gian huấn luyện thường cao hơn và khó giải thích hơn.
Mô hình được chọn nên cân bằng giữa Recall/F1 của lớp Attack, thời gian dự đoán và khả năng giải thích.