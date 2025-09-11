# 차량번호 OCR 프로그램

## 프로젝트 개요
이 프로젝트는 한국 차량 번호판을 자동으로 인식하는 OCR(Optical Character Recognition) 프로그램입니다.  
YOLOv8s를 이용한 차량 및 번호판 탐지, OpenCV를 이용한 이미지 보정, EasyOCR을 이용한 텍스트 인식을 통합하여 한국 차량 번호판을 효과적으로 인식합니다.

## 주요 기능

### 🎯 **핵심 기능**
- **다중 감지 모드**: 차량→번호판, 직접 번호판, 자동 감지 모드
- **한국 번호판 9가지 타입 분류**: 일반자가용, 영업용, 전기차, 외교관용, 군용, 건설기계, 이륜차, 임시운행, 특수용도
- **고급 이미지 전처리**: 슈퍼해상도, 지능형 디블러링, 적응형 대비 향상, 조명 정규화
- **한국어 특화 OCR**: 받침 없는 한글 처리, 형식 검증, 타입별 후처리

### 🚀 **성능 최적화**
- **시스템 자동 최적화**: CPU/GPU 자동 감지 및 성능 조정
- **한국 번호판 특화 YOLO**: 4:1 종횡비 최적화, 색상 특성 반영
- **실시간 성능 모니터링**: FPS, 메모리 사용량, 추론 시간 추적
- **로컬 환경 최적화**: 오프라인 동작, 클라우드 의존성 없음

### 🎨 **사용자 인터페이스**
- **Streamlit 웹 UI**: 직관적이고 사용자 친화적
- **다양한 입력 방식**: 파일 업로드, 카메라 촬영, 비디오 처리
- **상세 정보 표시**: 번호판 타입, 신뢰도, 유효성 검증 결과
- **성능 대시보드**: 실시간 처리 성능 및 시스템 정보

### 📊 **분석 및 출력**  
- **색상 기반 자동 분류**: K-means 클러스터링으로 정확한 색상 분석
- **유효성 검증**: 번호판 형식 자동 검증 및 오류 표시
- **통계 정보**: 처리 성능, 인식 정확도, 시스템 최적화 현황
- **결과 시각화**: 바운딩 박스, 분류 결과, 신뢰도 표시  

## 설치 방법

1. 저장소 클론  
   ```bash
   git clone https://github.com/yunchocopie/license-plate-ocr-project.git
   cd license-plate-ocr-project
   ```

2. 필요한 패키지 설치

   ### Windows

   ```bash
   python -m venv .venv
   source .venv/Scripts/activate
   pip install -r requirements.txt
   ```

   ### Unix (macOS/Linux/WSL)

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## 사전 설정

1. **모델 다운로드**

   ```bash
   source .venv/bin/activate    # Unix
   # 또는
   source .venv/Scripts/activate # Windows

   python download_models.py
   ```

   * 이 스크립트가 YOLO와 EasyOCR용 사전 학습 모델 파일을 내려받아 `models/` 폴더에 저장합니다.

## 사용 방법

1. 가상환경 활성화

   ### Windows

   ```bash
   source .venv/Scripts/activate
   ```

   ### Unix (macOS/Linux/WSL)

   ```bash
   source .venv/bin/activate
   ```

2. Streamlit 앱 실행

   ```bash
   streamlit run app_debug.py
   ```

3. 웹 브라우저에서 `http://localhost:8501` 열기

4. 이미지 업로드 또는 카메라로 차량 촬영

5. 결과 확인

## 개발자 정보

* **팀명:** 번호뭔지알려조
* **팀원:** 최윤정(팀장), 유우림, 김기윤, 이주환
* **개발기간:** 2025.03 ~

## 기술 스택

* **차량 탐지:** YOLOv8s
* **번호판 탐지:** YOLOv8s (custom trained)
* **번호판 보정:** OpenCV
* **번호판 숫자 인식:** EasyOCR
* **UI:** Streamlit

## 협력 및 후원

이 프로젝트는 사하구청과 협력하여 개발되었으며, 동아대학교 SW중심대학사업의 실증적 SW/AI 프로젝트로 진행되었습니다.
