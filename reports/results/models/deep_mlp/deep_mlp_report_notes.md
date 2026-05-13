# Deep MLP Report Notes

## 1. Mục tiêu

Mô hình Deep MLP được sử dụng làm mô hình Deep Learning cho bài toán phát hiện xâm nhập mạng.
Dữ liệu UNSW-NB15 sau tiền xử lý được biểu diễn dưới dạng vector đặc trưng, vì vậy kiến trúc MLP phù hợp hơn CNN/LSTM trong phạm vi hiện tại.

## 2. Cấu hình mô hình

Model: Deep Neural Network - Multi-Layer Perceptron
Kiến trúc:
- Dense 256 + ReLU + BatchNormalization + Dropout 0.30
- Dense 128 + ReLU + BatchNormalization + Dropout 0.30
- Dense 64 + ReLU + Dropout 0.20
- Dense 1 + Sigmoid

Optimizer: Adam
Learning rate: 0.001
Loss: binary_crossentropy
Epochs: 40
Batch size: 512
Validation split: 0.2
Class weight: balanced

Class weight được sử dụng để giảm ảnh hưởng của mất cân bằng dữ liệu.
EarlyStopping và ReduceLROnPlateau được sử dụng để hạn chế overfitting và điều chỉnh learning rate khi validation loss không cải thiện.

## 3. Kết quả đánh giá

Accuracy: 0.9063
Precision Attack: 0.8936
Recall Attack: 0.9420
F1-score Attack: 0.9172
ROC-AUC: 0.9794
PR-AUC: 0.9848

Trong bài toán IDS, Recall của lớp Attack là chỉ số rất quan trọng vì bỏ sót tấn công có thể gây rủi ro bảo mật.

## 4. Ma trận nhầm lẫn

True Negative: 31918
False Positive: 5082
False Negative: 2631
True Positive: 42701

False Negative là các mẫu tấn công bị dự đoán nhầm thành bình thường, cần được chú ý khi đánh giá hệ thống IDS.

## 5. Biểu đồ đã xuất

Các biểu đồ được lưu tại:

reports/figures/models/deep_mlp/

Bao gồm:

- Training loss curve
- Training accuracy curve
- Training precision/recall curve
- Training AUC curve
- Confusion matrix
- Normalized confusion matrix
- ROC curve
- Precision-Recall curve
- Probability distribution
- Threshold analysis
- Prediction distribution donut chart
- Metrics summary table
- Approximate feature importance from input weights

## 6. Nhận xét

Deep MLP có khả năng học quan hệ phi tuyến giữa các đặc trưng tốt hơn Logistic Regression.
So với Decision Tree đơn lẻ, MLP có thể biểu diễn quan hệ phức tạp hơn nhưng khó giải thích trực tiếp hơn.
Kết quả của Deep MLP được dùng để so sánh với hai baseline Logistic Regression và Decision Tree.