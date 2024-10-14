# Nhận Dạng Bất Thường

Dự án này thực hiện nhận dạng bất thường trong dữ liệu hình ảnh bằng cách sử dụng các mô hình học sâu. Nó sử dụng mạng VAE (Variational Autoencoder) kết hợp với Unet và lý thuyết giá trị cực trị (EVT) để phát hiện bất thường trong các hình ảnh y tế.

## Cấu Trúc Thư Mục
- /Data: Tải dữ liệu và tiền xử lý.
- /Model: Các mô hình và kiến trúc.
- /OpenSet: Nhận diện bất thường (EVT).
- /Training: Các tệp huấn luyện mô hình.
- /Utils: Các tiện ích hỗ trợ.
- main.py: Tệp chính để chạy chương trình.

## Cài Đặt

1. Clone hoặc tải về dự án này:
   ```bash
   https://github.com/taitruong256/Abnormal-detection.git
   ```

2. Cài đặt các thư viện cần thiết bằng pip:
    ```bash
    pip install -r requirements.txt
    ```
3. Để chạy chương trình, sử dụng lệnh sau trong terminal:

    ```bash
    python main.py
    ```

## Các Tham Số Có Thể Điều Chỉnh
Bạn có thể điều chỉnh một số tham số trong tệp main.py để phù hợp với nhu cầu của mình, chẳng hạn như:

IS_TRAIN: Xác định chế độ huấn luyện (True) hay đánh giá (False).

LATENT_DIM: Kích thước không gian tiềm ẩn của mô hình (số chiều vector tiềm ẩn).

NUM_EPOCHS: Số vòng lặp qua toàn bộ tập dữ liệu trong quá trình huấn luyện.

BETA: Hệ số điều chỉnh cho phần mất mát KL trong VAE.

LAMBDA_: Hệ số điều chỉnh cho một số thành phần mất mát phi tập trung. 

LEARNING_RATE: Tốc độ học của mô hình.

BATCH_SIZE: Số lượng mẫu trong mỗi batch dữ liệu khi huấn luyện.

VARIANCE: Phương sai của cho phân phối chuẩn trong không gian tiềm ẩn.

NUM_CLASSES: Số lượng lớp. 

INPUT_SHAPE: Kích thước đầu vào cho mô hình (chiều cao = chiều rộng = INPUT_SHAPE).

NUM_EXAMPLES: Tổng số ví dụ trong tập dữ liệu.

TAIL_SIZE: Kích thước đuôi (dùng để huấn luyện Weibull model xác định bất thường).

OMEGA_T: Ngưỡng xác suất nhận dạng bất thường. Nếu một mẫu có xác suất thuộc ngoài tập huấn luyện \leq OMEGA_T thì được xác định là bình thường. Ngược lại là bất thường. 

## Ghi Chú
Đảm bảo rằng bạn đã cấu hình đúng đường dẫn tới dữ liệu nếu có yêu cầu.

Để tải mô hình tốt nhất, bạn cần đảm bảo rằng tệp mô hình đã được lưu trữ.