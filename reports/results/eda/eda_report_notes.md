# EDA Report Notes

## 1. Tổng quan dữ liệu

Tập train có 175,341 dòng và 45 cột.
Tập test có 82,332 dòng và 45 cột.

## 2. Nhãn phân loại

Bài toán hiện tại sử dụng cột `label` cho phân loại nhị phân:

- `0`: Normal
- `1`: Attack

Phân phối nhãn trong tập train:

- Normal: 56,000
- Attack: 119,341

## 3. Thuộc tính phân loại và thuộc tính số

Số cột phân loại: 4
Các cột phân loại: ['proto', 'service', 'state', 'attack_cat']

Số cột số: 41

Các cột phân loại như `proto`, `service`, `state` cần được mã hóa trước khi train model.
Các cột số có thang đo khác nhau nên cần chuẩn hóa nếu sử dụng các mô hình nhạy với scale như Logistic Regression.

## 4. Biểu đồ đã xuất

Các biểu đồ EDA được lưu trong:

reports/figures/eda/

Các bảng kết quả EDA được lưu trong:

reports/results/eda/

## 5. Nhận xét dùng cho báo cáo

- Dataset phù hợp cho bài toán phát hiện xâm nhập mạng với nhãn Normal/Attack.
- Cột `attack_cat` có thể dùng để phân tích loại tấn công hoặc mở rộng sang bài toán phân loại đa lớp.
- Phân phối giữa các loại tấn công có thể không đồng đều, do đó khi đánh giá mô hình nên ưu tiên Recall và F1-score của lớp Attack.
- Các cột phân loại cần được One-Hot Encoding hoặc encoding phù hợp trước khi huấn luyện.
- Một số đặc trưng số có phân phối lệch mạnh, vì vậy việc trực quan hóa bằng log scale giúp dễ quan sát hơn.