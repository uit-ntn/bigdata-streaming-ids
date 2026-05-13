# Preprocessing Report Notes

## 1. Mục tiêu tiền xử lý

Dữ liệu UNSW-NB15 bao gồm cả thuộc tính số và thuộc tính phân loại.
Vì các mô hình học máy không thể sử dụng trực tiếp dữ liệu dạng chuỗi,
dữ liệu cần được chuyển đổi về dạng số trước khi huấn luyện.

## 2. Các cột bị loại bỏ

Các cột bị loại bỏ: ['id', 'attack_cat', 'label']

`id` là mã định danh dòng dữ liệu nên không có ý nghĩa dự đoán.
`attack_cat` thể hiện loại tấn công cụ thể, nên không được dùng trong bài toán phân loại nhị phân Normal/Attack để tránh rò rỉ nhãn.

## 3. Tách đặc trưng và nhãn

Cột nhãn sử dụng là `label`:

- 0 = Normal
- 1 = Attack

Phân phối nhãn trong tập train:

- Normal: 56,000
- Attack: 119,341

## 4. Xử lý thuộc tính số

Số lượng thuộc tính số: 39

Các thuộc tính số được xử lý bằng:

- SimpleImputer(strategy='median')
- StandardScaler()

Median được dùng để điền missing value vì bền hơn mean khi dữ liệu có outlier.
StandardScaler giúp đưa các đặc trưng số về cùng thang đo, đặc biệt cần thiết với các mô hình như Logistic Regression.

## 5. Xử lý thuộc tính phân loại

Số lượng thuộc tính phân loại: 3
Các cột phân loại: ['proto', 'service', 'state']

Các thuộc tính phân loại được xử lý bằng:

- SimpleImputer(strategy='most_frequent')
- OneHotEncoder(handle_unknown='ignore')

OneHotEncoder chuyển các cột dạng chuỗi như `proto`, `service`, `state` thành vector số.
Tham số `handle_unknown='ignore'` giúp hệ thống không lỗi nếu dữ liệu test hoặc dữ liệu streaming có category mới.

## 6. Số feature trước và sau tiền xử lý

Số feature trước tiền xử lý: 42
Số feature sau tiền xử lý: 194

Số feature tăng lên sau tiền xử lý do các thuộc tính phân loại được mở rộng bằng One-Hot Encoding.

## 7. Tránh data leakage

Preprocessor chỉ được fit trên tập train.
Tập test chỉ được transform bằng preprocessor đã fit.
Cách làm này giúp tránh data leakage, tức là tránh việc thông tin từ tập test bị sử dụng trong quá trình huấn luyện.

## 8. Tái sử dụng trong streaming

Preprocessor được lưu tại:

models/preprocessor.joblib

Trong giai đoạn mô phỏng streaming, các micro-batch mới sẽ phải đi qua đúng preprocessor này trước khi đưa vào mô hình dự đoán.

## 9. Output

Dữ liệu processed được lưu trong:

data/processed/

Biểu đồ preprocessing được lưu trong:

reports/figures/preprocessing/

Bảng kết quả preprocessing được lưu trong:

reports/results/preprocessing/

## 10. Các loại biểu đồ đã tạo

- Flow diagram: mô tả luồng tiền xử lý.
- Donut chart: tỷ trọng nhóm feature và độ thưa của ma trận sau xử lý.
- Slope chart: số feature trước và sau tiền xử lý.
- Lollipop chart: cardinality và số feature sau One-Hot Encoding.
- Missing-value matrix: trực quan tình trạng missing value.
- Histogram, density line, violin plot, ECDF: so sánh dữ liệu số trước và sau scaling.
- Summary table image: bảng tóm tắt tiền xử lý.