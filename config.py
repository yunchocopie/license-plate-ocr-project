import os
from pathlib import Path

"""
차량번호 OCR 프로그램의 설정 파일

이 파일은 다음과 같은 설정 및 파라미터를 정의합니다:
1. 파일 경로 설정 (모델 가중치, 데이터 디렉토리 등)
2. 모델 파라미터 설정 (YOLOv8s, EasyOCR 등)
3. 이미지 처리 파라미터 (크기, 임계값 등)
4. UI 관련 설정
"""
# 기본 경로 설정
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# 모델 파일 경로
VEHICLE_DETECTION_MODEL = os.path.join(MODEL_DIR, "yolov8s.pt")  # 기본 YOLOv8s 모델 (차량 탐지용)
PLATE_DETECTION_MODEL = os.path.join(MODEL_DIR, "license_plate_detection.pt")  # 학습된 번호판 탐지 모델

# 모델 파라미터
VEHICLE_DETECTION_CONF = 0.25  # 차량 탐지 신뢰도 임계값
PLATE_DETECTION_CONF = 0.5     # 번호판 탐지 신뢰도 임계값 (license-plate-ocr-project 기반으로 조정)

# 이미지 처리 파라미터
IMAGE_SIZE = (640, 640)        # YOLO 입력 이미지 크기
PLATE_SIZE = (240, 80)         # 번호판 정규화 크기
BLUR_KERNEL_SIZE = (5, 5)      # 블러 커널 크기
BLUR_SIGMA = 1.0               # 가우시안 블러 시그마 값

# OCR 설정
OCR_LANGUAGES = ['ko']         # 한국어와 인식
OCR_GPU = True                 # GPU 사용 여부 (가능한 경우)
OCR_ALLOWED_CHARS = '가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히' + '0123456789'  # 허용된 문자

# UI 설정
STREAMLIT_PAGE_TITLE = "차량번호 OCR 프로그램"
STREAMLIT_PAGE_ICON = "🚗"
STREAMLIT_LAYOUT = "wide"