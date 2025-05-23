# 차량번호 OCR 프로그램 분석 보고서

## 프로젝트 개요
이 프로젝트는 한국 차량 번호판을 자동으로 인식하는 OCR(Optical Character Recognition) 프로그램입니다. 사하구청에서 요청한 프로젝트로, 개인 PC에서 쉽게 사용할 수 있고 CPU 및 GPU 환경 모두에서 작동하는 한국 차량 번호판 OCR 프로그램을 개발하는 것이 목적입니다.

## 개발 배경 및 필요성
- 기존 chatGPT와 같은 SaaS 기반 프로그램은 개인정보보호 문제로 차량 사진 입력이 어려움
- 오픈소스 OCR 프로그램은 외국 차량 번호판 인식률은 높지만, 한국 차량 번호판 인식 성능은 떨어지거나 유료인 경우가 많음
- 개인 PC에서 쉽게 사용할 수 있고, CPU 및 GPU 환경 모두에서 작동하는 한국 차량 번호판 OCR 프로그램이 필요

## 시스템 구성도 (아키텍처)
```
[입력 영상/이미지]
       ↓
[YOLOv8s 차량 탐지]
       ↓
[탐지된 차량 영역 Crop]
       ↓
[YOLOv8s 번호판 탐지 모델]
       ↓
[탐지된 번호판 영역 Crop]
       ↓
[OCR 엔진 (EasyOCR 또는 Tesseract)]
       ↓
[번호 추출 결과 출력]
```

## 프로젝트 구조
```
license-plate-ocr-project/
├── data/                   # 데이터 및 모델 파일 디렉토리
│   ├── models/             # 사전 훈련된 모델 파일 저장
│   │   ├── craft_mlt_25k.pth         # CRAFT 텍스트 감지 모델 (EasyOCR 사용)
│   │   ├── korean_g2.pth             # 한국어 인식 모델 (EasyOCR 사용)
│   │   ├── license_plate_detection.pt # 번호판 탐지 모델 (YOLOv8 기반 커스텀 모델)
│   │   └── yolov8s.pt                # 차량 탐지를 위한 YOLOv8s 모델
│   ├── processed/          # 전처리된 이미지 저장 디렉토리
│   └── raw/                # 원본 이미지 저장 디렉토리
│       ├── car.jpg         # 테스트용 차량 이미지
│       └── plate.jpg       # 테스트용 번호판 이미지
├── src/                    # 소스 코드
│   ├── detection/          # 차량 및 번호판 탐지 관련 코드
│   │   ├── __init__.py
│   │   ├── plate_detector.py  # 번호판 탐지 모듈
│   │   └── vehicle_detector.py # 차량 탐지 모듈
│   ├── ocr/                # OCR 엔진 관련 코드
│   │   ├── __init__.py
│   │   ├── ocr_engine.py   # EasyOCR 기반 OCR 엔진
│   │   └── text_postprocess.py # 인식된 텍스트 후처리
│   ├── preprocessing/      # 이미지 전처리 코드
│   │   ├── __init__.py
│   │   ├── blur_correction.py # 흐림 보정 모듈
│   │   ├── image_processor.py # 이미지 처리 통합 모듈
│   │   ├── normalize.py    # 이미지 비율 정규화
│   │   └── perspective.py  # 원근 왜곡 보정
│   └── utils/              # 유틸리티 함수
│       ├── __init__.py
│       ├── metrics.py      # 성능 측정 메트릭
│       └── visualization.py # 결과 시각화 도구
├── ui/                     # Streamlit UI 관련 코드
│   ├── pages/              # Streamlit 멀티페이지 앱 구현
│   │   ├── __init__.py
│   │   ├── analysis.py     # 인식 결과 분석 페이지
│   │   ├── home.py         # 홈 페이지
│   │   └── settings.py     # 설정 페이지
│   ├── __init__.py
│   └── components.py       # 재사용 가능한 UI 컴포넌트
├── app.py                  # 메인 Streamlit 애플리케이션
├── app_debug.py            # 디버깅용 애플리케이션
├── config.py               # 설정 파일
├── download_models.py      # 필요한 모델 파일 다운로드 스크립트
├── README.md               # 프로젝트 설명
└── requirements.txt        # 필요한 패키지 목록
```

## 주요 기능 및 구현 내용

### 1. 차량 감지 (`src/detection/vehicle_detector.py`)
- YOLOv8s 모델을 사용하여 이미지 또는 비디오에서 차량 감지
- COCO 데이터셋 기준으로 차량 클래스(자동차, 트럭, 버스)에 해당하는 객체만 필터링
- GPU 가속을 자동으로 감지하여 사용 가능한 환경에서 활용

### 2. 번호판 감지 (`src/detection/plate_detector.py`)
- 커스텀 학습된 YOLOv8s 모델을 사용하여 차량 이미지에서 번호판 영역 감지
- 감지된 번호판 영역을 추출하여 다음 단계로 전달

### 3. 이미지 전처리 (`src/preprocessing/`)
- `image_processor.py`: 다양한 전처리 기법을 통합적으로 적용하는 클래스
- `blur_correction.py`: 흐림 보정을 위한 다양한 알고리즘 구현 
  (언샤프 마스킹, 라플라시안 선명화, 경계선 개선 등)
- `perspective.py`: 기울기 및 원근 왜곡 보정 알고리즘 구현
  (윤곽선 검출, 회전 행렬 계산, 원근 변환 등)
- `normalize.py`: 번호판 이미지의 비율 정규화 및 테두리 제거 등 구현

### 4. OCR 인식 (`src/ocr/`)
- `ocr_engine.py`: EasyOCR을 활용한 텍스트 인식 엔진
  (한국어 및 영어 문자 인식, GPU 가속 지원)
- `text_postprocess.py`: 인식된 텍스트 후처리 클래스
  (유사 문자 교정, 한국 번호판 형식 포맷팅 등)

### 5. 유틸리티 (`src/utils/`)
- `visualization.py`: 감지 및 인식 결과 시각화 함수
  (차량 박스, 번호판 박스, OCR 결과 등 시각화)
- `metrics.py`: 성능 평가를 위한 메트릭 계산 함수
  (IOU, 정확도, 유사도, 실행 시간 등)

### 6. 웹 UI (`app.py`)
- Streamlit을 활용한 웹 기반 사용자 인터페이스
- 이미지 업로드, 카메라 촬영, 비디오 업로드 등의 입력 방식 지원
- 처리 결과 시각화 및 인식된 번호판 텍스트 표시

## 기술 스택
- **차량 탐지**: YOLOv8s (사전 학습된 모델)
- **번호판 탐지**: YOLOv8s (커스텀 학습된 모델)
- **번호판 보정**: OpenCV (흐림 보정, 기울기 보정, 비율 정규화)
- **번호판 숫자 인식**: EasyOCR
- **UI**: Streamlit

## YOLOv8s vs YOLOv8n 비교
- **YOLOv8n (Nano)**: 속도는 빠르지만, 작은 객체(번호판) 탐지 성능이 낮음
- **YOLOv8s (Small)**: 약간 느리지만 더 높은 정확도로 번호판 탐지 가능

## OpenCV vs EasyOCR 비교
- 차량 번호판은 한글 + 영어 + 숫자 조합이 많음 → EasyOCR이 더 강력함
- 다양한 글씨체, 조명 변화, 흐릿한 번호판에서도 EasyOCR이 더 잘 동작함

## 개발 현황
- 주간 보고서를 통해 확인해 본 결과, 팀은 현재 OCR 모델 선택 및 적용 단계와 번호판 영역 전처리 단계에 있습니다.
- 차량 및 번호판 탐지 모델 구현, 이미지 전처리 파이프라인 구축은 완료되었습니다.
- 코드 구조는 완성되어 있으며, 세부 모듈별 구현도 상당히 진행된 상태입니다.
- 다음 계획으로는 번호판 보정 및 문자 인식 단계별 성능 튜닝과 처리 속도 개선이 예정되어 있습니다.

## 개발팀 정보
- **팀명**: 번호뭔지알려조
- **팀원**: 최윤정(팀장), 유우림, 김기윤, 이주환
- **개발기간**: 2025.03 ~

## 결론
이 프로젝트는 차량 번호판 OCR을 위한 완전한 파이프라인을 구현하였으며, 실용적인 응용 프로그램으로 개발되고 있습니다. 특히 한국 번호판 인식에 특화된 후처리 로직이 인상적입니다. 모듈화된 설계로 각 컴포넌트를 쉽게 개선하거나 대체할 수 있는 유연한 구조를 가지고 있습니다.
