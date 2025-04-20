import os
import tempfile
from PIL import Image
import numpy as np
import cv2
from image_detection.find_plate import detect_plate
from Input.unzip import extract_zip_safely

def uploaded_files(uploaded_files, save_dir='cropped_plate'):
    os.makedirs(save_dir, exist_ok=True) # 결과를 저장할 폴더 생성
    temp_dir = tempfile.mkdtemp() # 임시 디렉터리 생(압축 해제 및 이미지 저장용)

    saved_files = [] # 처리 완료된 이미지 경로 리스트

    for file in uploaded_files:
        if file.name.endswith('.zip'): # zip 파일이면 임시 디렉터리에 저장한 뒤 압축 해제
            zip_path = os.path.join(temp_dir, file.name)
            with open(zip_path, 'wb') as f:
                f.write(file.read())
            extract_zip_safely(zip_path, temp_dir)
        else: # 일반이미지 파일은 그냥 임시 폴더에 저장
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.read())

    for root, dirs, files in os.walk(temp_dir): # 임시 디렉터리내 모든 파일 반복
        for fname in files:
            image_path = os.path.join(root, fname)
            if not is_valid_image_file(fname): # 유효하지 않은 파일 건너 뜀
                continue
            if not os.path.isfile(image_path): # 파일이 아닌경우 패스
                continue
            try:
                image = Image.open(image_path).convert("RGB") # 이미지 열기 + RGB 로 변환
            except Exception as e:
                print(f"❌ 열기 실패: {image_path} → {e}")
                continue

            img_np = np.array(image) # numpy 배열로 변환
            cropped = detect_plate(img_np) # 번호판 감지 + 잘라내기

            if cropped is not None:
                # 잘라낸 번호판을 저장
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
