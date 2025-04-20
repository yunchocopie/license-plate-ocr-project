import os
import tempfile
from PIL import Image
import numpy as np
import cv2
from image_detection.find_plate import detect_plate
from Input.unzip import extract_zip_safely

def uploaded_files(uploaded_files, save_dir='cropped_plate'):
    os.makedirs(save_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()

    saved_files = []

    for file in uploaded_files:
        if file.name.endswith('.zip'):
            zip_path = os.path.join(temp_dir, file.name)
            with open(zip_path, 'wb') as f:
                f.write(file.read())
            extract_zip_safely(zip_path, temp_dir)
        else:
            with open(zip_path, "wb") as f:
                f.write(file.read())

    for root, dirs, files in os.walk(temp_dir):
        for fname in files:
            image_path = os.path.join(root, fname)
            if not is_valid_image_file(fname):
                continue
            if not os.path.isfile(image_path): # 파일이 아닌경우 패스
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"❌ 열기 실패: {image_path} → {e}")
                continue

            img_np = np.array(image)
            cropped = detect_plate(img_np)

            if cropped is not None:
                save_name = os.path.splitext(fname)[0] + "_plate.jpg"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, cropped)
                saved_files.append(save_path)

    return saved_files  # 저장된 파일 경로 리스트 반환

def is_valid_image_file(fname):
    # macOS 메타 파일 제외
    return (
        isinstance(fname, str) and
        not fname.startswith("._") and
        not fname.endswith(".DS_Store") and
        fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
