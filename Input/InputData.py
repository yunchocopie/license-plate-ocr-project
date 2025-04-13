import zipfile
import io
from PIL import Image
import numpy as np

images = []
def inputData(upload_images):
    for upload_image in upload_images:
        file_name = upload_image.name.lower()

        # 압축 파일인 경우 따로 처리
        if file_name.endswith('.zip'):
            try:
                zip_bytes = io.BytesIO(upload_image.read()) # zip 바이트 데이터를 메모리상의 파일로 변경
                with zipfile.ZipFile(zip_bytes, 'r') as zip_ref: # zip 파일을 읽기 모드로 열기
                    image_list = zip_ref.namelist() # zip 파일안 목록 확인
                    for image in image_list:
                        # macOS 메타파일 스킵
                        if image.startswith("__MACOSX") or "/._" in image or image.startswith("._"):
                            continue

                        if image.lower().endswith(('.jpg', '.jpeg', '.png')):
                            with zip_ref.open(image) as img_file: # zip 파일안에 있는 개별파일을 메모리에서 읽기 전용으로 열기
                                img = Image.open(img_file)
                                array = np.array(img)
                                images.append(array)
            except Exception as e:
                print("ZIP 파일 처리 실패: ", e)
        else:
            try:
                img = Image.open(upload_image)
                array = np.array(img)
                images.append(array)
            except Exception as e:
                print("이미지 파일 처리 실패")