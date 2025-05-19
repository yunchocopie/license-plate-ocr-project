# 차량번호 OCR 프로그램

## 프로젝트 개요
이 프로젝트는 한국 차량 번호판을 자동으로 인식하는 OCR(Optical Character Recognition) 프로그램입니다. YOLOv8s를 이용한 차량 및 번호판 탐지, OpenCV를 이용한 이미지 보정, 그리고 EasyOCR을 이용한 텍스트 인식을 통합하여 한국 차량 번호판을 효과적으로 인식합니다.

## 주요 기능
- 이미지나 비디오에서 차량 자동 감지
- 차량에서 번호판 영역 자동 감지
- 번호판 이미지 전처리 (흐림 보정, 기울기 보정, 비율 정규화)
- 번호판 텍스트 인식 및 추출
- Streamlit 기반의 사용자 친화적 웹 UI
- 로컬 환경에서 CPU 및 GPU 모두 지원

## 설치 방법
1. 저장소 클론
```bash
git clone https://github.com/yunchocopie/license-plate-ocr-project.git
cd license-plate-ocr-project
```

2. 필요한 패키지 설치
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## 사용 방법
1. 프로그램 실행
```bash
source .venv/Scripts/activate
streamlit run app_debug.py
```

2. 웹 브라우저에서 `http://localhost:8501` 열기

3. 이미지 업로드 또는 카메라로 차량 촬영

4. 결과 확인

## 개발자 정보
- 팀명: 번호뭔지알려조
- 팀원: 최윤정(팀장), 유우림, 김기윤, 이주환
- 개발기간: 2025.03 ~ 2025.05

## 기술 스택
- 차량 탐지: YOLOv8s
- 번호판 탐지: YOLOv8s (custom trained)
- 번호판 보정: OpenCV
- 번호판 숫자 인식: EasyOCR
- UI: Streamlit

## 감사의 말
이 프로젝트는 사하구청과 협력하여 개발되었으며, 동아대학교 SW중심대학사업의 실증적 SW/AI 프로젝트로 진행되었습니다.
