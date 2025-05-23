# OCR Engine 관련
OCR_LANGUAGES = ['ko', 'en']
OCR_GPU = True  # 사용 가능하면 True, 아니면 False
# '루'를 포함하여 번호판에 사용될 수 있는 모든 문자 포함
OCR_ALLOWED_CHARS = '0123456789가나다라마바사아자차카타파하거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호배육해공루' # <--- '루' 추가 확인!
MODEL_DIR = './easyocr_models'  # EasyOCR 모델 저장 경로
DOWNLOAD_ENABLED = True # 모델 자동 다운로드 여부

# ImageProcessor 관련
# PLATE_SIZE: (width, height) EasyOCR이 잘 인식할 수 있는 적절한 크기. 너무 작으면 안됨.
# 예: 구형 번호판 비율 (약 2:1 ~ 3:1), 신형 번호판 비율 (약 4:1 ~ 5:1)
PLATE_SIZE = (320, 80)  # 예시 크기 (너비, 높이) - 테스트하며 조절
PLATE_TEXT_COLOR = 'bright' # 번호판 글자색 ('bright' on dark background, or 'dark' on bright background) - Perspective 이진화에 영향

# BlurCorrection 관련
BLUR_KERNEL_SIZE = (5, 5) # 가우시안 블러 커널 크기
BLUR_SIGMA = 0            # 가우시안 블러 시그마 (0이면 커널 크기에 맞게 자동 계산)