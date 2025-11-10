# 번호판 OCR 테스트 스위트

## 개요

한국 번호판 OCR 프로젝트의 통합 테스트 모음입니다. 전처리, OCR, 리그레션 테스트를 포함합니다.

## 테스트 파일 목록

### 1. 리그레션 테스트
- **`test_plate_regression.py`**: 전체 OCR 파이프라인 성능 검증
- **`test_adaptive_preprocessing.py`**: 적응형 전처리 알고리즘 테스트
- **`test_slot_pipeline.py`**: 슬롯 기반 파이프라인 테스트

### 2. 전처리 테스트
- **`test_advanced_preprocessing.py`**: 고급 전처리 기능 (슈퍼해상도, 디블러링 등)
- **`test_preprocessing_only.py`**: OCR 없이 전처리 파이프라인만 검증
- **`test_background_removal.py`**: 배경 제거 알고리즘 테스트

### 3. 개별 기능 테스트
- **`test_char_segmentation.py`**: 문자 영역 추출 및 분리 테스트
- **`test_uploaded_plate.py`**: 업로드된 번호판 이미지의 전처리 단계 시각화

## 테스트 구조

```
tests/
  test_plate_regression.py      # 리그레션 테스트
  test_adaptive_preprocessing.py # 적응형 전처리
  test_slot_pipeline.py          # 슬롯 파이프라인
  test_advanced_preprocessing.py # 고급 전처리
  test_preprocessing_only.py     # 전처리 전용
  test_background_removal.py     # 배경 제거
  test_char_segmentation.py      # 문자 분리
  test_uploaded_plate.py         # 업로드 이미지 디버깅
  README.md                      # 이 문서

data/test/
  general/         # 일반 자가용
    labels.json    # Ground Truth 데이터
    *.jpg, *.png   # 테스트 이미지
  commercial/      # 영업용
  electric/        # 전기차
  diplomatic/      # 외교관용
  military/        # 군용
  construction/    # 건설기계
  motorcycle/      # 이륜차
  temporary/       # 임시운행
  special/         # 특수용도
```

## Ground Truth 형식

`labels.json` 파일은 다음 형식을 따릅니다:

```json
{
  "image_filename.jpg": {
    "plate_count": 1,
    "text": "12가3456",
    "boxes": [[x1, y1, x2, y2]]
  }
}
```

- **plate_count**: 이미지에 포함된 번호판 개수
- **text**: 예상되는 OCR 결과 텍스트
- **boxes**: 번호판 바운딩 박스 좌표 (선택사항)

## 테스트 실행

### 전체 테스트 실행

```bash
pytest tests/test_plate_regression.py -v -s
```

### 특정 유형만 테스트

```bash
pytest tests/test_plate_regression.py -v -s -k "general"
```

### 검출 수만 테스트

```bash
pytest tests/test_plate_regression.py -v -s -k "detection_count"
```

### OCR 정확도만 테스트

```bash
pytest tests/test_plate_regression.py -v -s -k "ocr_accuracy"
```

## 성공 기준

- **검출 Recall**: GT 대비 ±30% 범위 내
- **OCR 정확도**: 최소 50% 이상
- **False Positive**: 30% 이상 감소 (Task 1B 완료 후)

## 데이터 준비

각 번호판 유형별로 최소 10~20장의 테스트 이미지를 준비하고, `labels.json`에 GT 데이터를 추가하세요.

### 예시: 일반 자가용 데이터 준비

1. `data/test/general/` 폴더에 이미지 추가
2. `data/test/general/labels.json` 파일에 GT 데이터 추가

```json
{
  "car_001.jpg": {
    "plate_count": 1,
    "text": "12가3456"
  },
  "car_002.jpg": {
    "plate_count": 1,
    "text": "301허1234"
  }
}
```

## 테스트 결과 해석

테스트 실행 후 다음 정보가 출력됩니다:

```
일반 OCR 정확도: 75.00% (15/20)
영업용 OCR 정확도: 60.00% (12/20)
...

==================================================
전체 성능 요약
==================================================
총 이미지 수: 100
총 검출 수: 98
정확한 OCR: 72
전체 정확도: 72.00%

유형별 상세:
  general: 75.00% (15/20)
  commercial: 60.00% (12/20)
  ...
==================================================
```

## 테스트 실행 방법

### pytest를 사용한 자동화 테스트
```bash
# 전체 테스트 실행
pytest tests/ -v -s

# 특정 테스트 파일 실행
pytest tests/test_plate_regression.py -v -s

# 특정 유형만 테스트
pytest tests/test_plate_regression.py -v -s -k "general"
```

### 개별 스크립트 실행
```bash
# 전처리 파이프라인 테스트
python tests/test_preprocessing_only.py

# 업로드 이미지 디버깅
python tests/test_uploaded_plate.py

# 고급 전처리 테스트
python tests/test_advanced_preprocessing.py
```

## 주의사항

- 테스트 이미지는 실제 사용 환경과 유사한 조건으로 준비
- 다양한 조명, 각도, 거리의 이미지 포함
- GT 데이터는 정확하게 입력 (오타 주의)
- 테스트 실행 시 모델 로딩으로 인한 초기 지연 발생 가능
- 시각화 테스트는 `cv2.imshow()` 창을 닫아야 다음 단계로 진행됨
