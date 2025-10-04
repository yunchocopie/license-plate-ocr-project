import os
from pathlib import Path

"""
차량번호 OCR 프로그램의 통합 설정 파일

이 파일은 모든 모듈의 설정 및 파라미터를 통합 관리합니다:
1. 파일 경로 설정 (모델 가중치, 데이터 디렉토리 등)
2. 모델 파라미터 설정 (YOLOv8s, EasyOCR 등)
3. 이미지 처리 파라미터 (크기, 임계값 등)
4. OCR 및 후처리 설정
"""

# ==========================================
# 기본 경로 설정
# ==========================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
DEBUG_DIR = os.path.join(DATA_DIR, "debug")  # 디버그 이미지 저장 경로

# 디버그 디렉토리가 없으면 생성
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR, exist_ok=True)

# ==========================================
# 모델 파일 경로 설정
# ==========================================
VEHICLE_DETECTION_MODEL = os.path.join(MODEL_DIR, "yolov8s.pt")  # 기본 YOLOv8s 모델 (차량 탐지용)
PLATE_DETECTION_MODEL = os.path.join(MODEL_DIR, "license_plate_detection.pt")  # 학습된 번호판 탐지 모델

# ==========================================
# 차량/번호판 탐지 모델 파라미터
# ==========================================
VEHICLE_DETECTION_CONF = 0.25  # 차량 탐지 신뢰도 임계값
PLATE_DETECTION_CONF = 0.01    # 번호판 탐지 신뢰도 임계값 (낮은 값으로 더 많은 감지 허용)

# YOLO 모델 설정
IMAGE_SIZE = (640, 640)        # YOLO 입력 이미지 크기

# ==========================================
# 이미지 전처리 파라미터
# ==========================================
# 번호판 타입별 최적화된 크기 설정 (width, height)
PLATE_SIZES = {
    'general': (320, 80),           # 일반 번호판 (03마7893, 04루3284) - 가장 일반적
    'general_3digit': (360, 80),    # 3자리 번호판 (145하1937)
    'commercial': (400, 80),        # 영업용 번호판 (경기37바2120)
    'electric': (320, 80),          # 전기차 (일반과 동일)
    'diplomatic': (350, 80),        # 외교관용
    'military': (300, 80),          # 군용
    'construction': (400, 80),      # 건설기계
    'motorcycle': (280, 60),        # 이륜차 (작은 크기)
    'temporary': (320, 80),         # 임시운행
    'default': (320, 80)            # 기본값
}

PLATE_SIZE = PLATE_SIZES['general']  # 하위 호환성을 위한 기본값
BLUR_KERNEL_SIZE = (5, 5)      # 블러 커널 크기
BLUR_SIGMA = 0                 # 가우시안 블러 시그마 값 (0: 커널 크기에 맞게 자동 계산)

# 한국 번호판 최적화 설정
KOREAN_PLATE_OPTIMIZATION = {
    'min_char_width': 15,           # 최소 문자 폭 (픽셀)
    'min_char_height': 25,          # 최소 문자 높이 (픽셀)
    'char_spacing_threshold': 8,    # 문자 간격 임계값
    'line_height_ratio': 0.7,       # 라인 높이 비율
    'aspect_ratio_tolerance': 0.3    # 종횡비 허용 오차
}

# ==========================================
# OCR 엔진 설정
# ==========================================
OCR_LANGUAGES = ['ko', 'en']         # 인식 언어 (한국어 + 영어 지원)
OCR_GPU = True                        # GPU 사용 여부 (가능한 경우)
# 한국 번호판에서 사용되는 모든 한글 문자
# 일반 번호판: 가나다라마바사아자차카타파하 + 거너더러머버서어저처커터퍼허 + 고노도로모보소오조초코토포호 + 구누두루무부수우주추쿠투푸후 + 그느드르므브스으즈츠크트프흐 + 기니디리미비시이지치키티피히
# 지역명: 서울,부산,대구,인천,광주,대전,울산,세종,경기,강원,충북,충남,전북,전남,경북,경남,제주
# 특수용도: 국,육,해,공,합 (군용), 외교,영사 (외교관용) 등
OCR_ALLOWED_CHARS = '가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히경광국도로마배부산서인전제충울세종원북남제주외교영사국육해공합' + '0123456789'

# EasyOCR 모델 설정
DOWNLOAD_ENABLED = False              # 로컬 모델 사용 (download_models.py로 사전 다운로드된 모델 사용)

# OCR 결과 신뢰도 임계값
MIN_OCR_CONFIDENCE = 0.1              # 최소 OCR 신뢰도 (낮게 설정하여 더 많은 결과 허용)

# ==========================================
# 후처리 설정
# ==========================================
# 한국어 번호판 지역명
KOREAN_REGIONS = [
    '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
    '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'
]
