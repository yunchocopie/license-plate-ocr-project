# 번호판 OCR 리그레션 테스트

## 개요

1차 작업(Task 1A~1D) 완료 후 성능 검증을 위한 리그레션 테스트입니다.

## 테스트 구조

```
tests/
  test_plate_regression.py  # 리그레션 테스트 스크립트

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

## 주의사항

- 테스트 이미지는 실제 사용 환경과 유사한 조건으로 준비
- 다양한 조명, 각도, 거리의 이미지 포함
- GT 데이터는 정확하게 입력 (오타 주의)
- 테스트 실행 시 모델 로딩으로 인한 초기 지연 발생 가능

## 추가 테스트

더 상세한 테스트가 필요한 경우:

1. `test_char_segmentation.py`: 문자 영역 추출 테스트
2. `test_advanced_preprocessing.py`: 고급 전처리 테스트
3. `test_background_removal.py`: 배경 제거 테스트
