import os 
import cv2 
import pydicom 
import pylibjpeg 
import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from joblib import Parallel, delayed

pydicom.config.image_handlers = ['pylibjpeg']

def read_and_crop_image(output_dir, dicom_path, patient_id, image_id):
    try:
        # Đọc file DICOM
        dicom_data = pydicom.dcmread(dicom_path)
        img_array = dicom_data.pixel_array
        if dicom_data.PhotometricInterpretation == "MONOCHROME1":
            img_array = 1 - img_array

        # Chuyển đổi sang định dạng 8-bit (0-255)
        img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        img_8bit = (img_normalized * 255).astype(np.uint8)

        # Áp dụng cân bằng histogram
        img_8bit = cv2.equalizeHist(img_8bit)

        # Áp dụng threshold để tìm contour
        _, binary_img = cv2.threshold(img_8bit, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Tìm contour có diện tích lớn nhất
        max_contour = max(contours, key=cv2.contourArea)

        # Tạo bounding box từ contour lớn nhất
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(binary_img, (x, y), (x+w, y+h), (255, 0, 0), 5)

        # Cắt ảnh dựa trên bounding box
        cropped_img = img_8bit[y:y+h, x:x+w]

        # Resize ảnh về kích thước 512x512
        final_img = cv2.resize(cropped_img, (512, 512), interpolation=cv2.INTER_AREA)

        # Chuẩn hóa ảnh (0-1)
        final_img = final_img / 255.0

        # Tạo thư mục con cho mỗi bệnh nhân nếu chưa tồn tại
        patient_dir = os.path.join(output_dir, str(patient_id))
        os.makedirs(patient_dir, exist_ok=True)

        # Lưu ảnh
        image_path = os.path.join(patient_dir, f"{image_id}.png")
        plt.imsave(image_path, final_img, cmap='gray')

    except Exception as e:
        print(f"Error processing image {dicom_path}: {e}")
        final_img = np.zeros((256, 256), dtype=np.float32)

    return final_img

def process_and_save_images(df, output_dir='/kaggle/working/train_images'):
    # Tạo thư mục gốc nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    def process_row(idx, row, total):
        return read_and_crop_image(output_dir, row['dicom_path'], row['patient_id'], row['image_id'])

    results = Parallel(n_jobs=-1)(delayed(process_row)(idx, row, len(df)) for idx, row in tqdm(df.iterrows(), total=len(df)))
