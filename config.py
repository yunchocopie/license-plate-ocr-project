import os
from pathlib import Path

"""
차량번호 OCR 프로그램의 설정 파일

이 파일은 다음과 같은 설정 및 파라미터를 정의합니다:
1. 파일 경로 설정 (모델 가중치, 데이터 디렉토리 등)
2. 모델 파라미터 설정 (YOLOv8s, EasyOCR 등)
3. 이미지 처리 파라미터 (크기, 임계값 등)
4. UI 관련 설정
5. 이미지 품질 평가 설정 (1단계 추가)
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


# ==========================================
# 이미지 품질 평가 설정
# ==========================================

# 품질 평가 임계값
QUALITY_THRESHOLDS = {
    # 선명도 임계값
    'sharpness': {
        'high': 800.0,      # 높은 선명도 (최소 처리)
        'medium': 200.0,    # 중간 선명도 (적당한 처리)
        'low': 50.0         # 낮은 선명도 (전체 처리)
    },
    
    # 대비도 임계값
    'contrast': {
        'high': 60.0,       # 높은 대비도
        'medium': 30.0,     # 중간 대비도
        'low': 10.0         # 낮은 대비도
    },
    
    # 노이즈 레벨 임계값 (낮을수록 좋음)
    'noise': {
        'low': 5.0,         # 낮은 노이즈 (좋음)
        'medium': 12.0,     # 중간 노이즈
        'high': 20.0        # 높은 노이즈 (나쁨)
    },
    
    # 종합 품질 점수 임계값 (0-100)
    'overall': {
        'excellent': 75.0,  # 우수한 품질 (최소 처리)
        'good': 45.0,       # 양호한 품질 (적당한 처리)
        'poor': 0.0         # 낮은 품질 (전체 처리)
    }
}

# 적응형 처리 모드 설정
ADAPTIVE_PROCESSING = {
    'enabled': True,            # 적응형 처리 활성화 여부
    'quality_weight': {         # 품질 지표별 가중치
        'sharpness': 0.35,      # 선명도 가중치 (가장 중요)
        'contrast': 0.25,       # 대비도 가중치
        'noise': 0.25,          # 노이즈 레벨 가중치
        'brightness': 0.15      # 밝기 가중치 (덜 중요)
    },
    'processing_levels': {      # 처리 수준별 설정
        'minimal': {            # 최소 처리 (고품질 이미지)
            'steps': ['normalize', 'enhance_minimal'],
            'description': '고품질 이미지 - 최소한의 전처리'
        },
        'moderate': {           # 적당한 처리 (중간 품질)
            'steps': ['denoise_light', 'blur_correction_light', 'normalize', 'enhance'],
            'description': '중간 품질 이미지 - 적당한 전처리'
        },
        'full': {               # 전체 처리 (저품질 이미지)
            'steps': ['denoise_heavy', 'blur_correction', 'perspective_correction', 'normalize', 'enhance_heavy'],
            'description': '저품질 이미지 - 전체 전처리'
        }
    }
}

# 이미지 품질 정규화 기준값
QUALITY_NORMALIZATION = {
    'sharpness_max': 1000.0,        # 선명도 최대값
    'contrast_max': 80.0,           # 대비도 최대값
    'noise_max': 20.0,              # 노이즈 최대값
    'brightness_optimal': 127.5,    # 최적 밝기값
    'brightness_tolerance': 50.0    # 밝기 허용 범위
}
