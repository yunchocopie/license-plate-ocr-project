# 번호판 텍스트 인식 모델 학습 가이드

## 개요

CRNN (Convolutional Recurrent Neural Network) 기반 한국 번호판 텍스트 인식 모델 학습 파이프라인입니다.

## 디렉토리 구조

```
training/
  augmentation.py            # 데이터 증강 모듈
  train_text_recognition.py  # 학습 스크립트
  README.md                  # 이 파일

data/
  train/                     # 학습 데이터
    images/                  # 학습 이미지
    labels.json              # 학습 레이블
  val/                       # 검증 데이터
    images/                  # 검증 이미지
    labels.json              # 검증 레이블
  augmented/                 # 증강 데이터 (자동 생성)

models/
  crnn_best.pth             # 학습된 모델 (자동 저장)
```

## 데이터 준비

### 1. 데이터 수집

다음 소스에서 한국 번호판 이미지를 수집:

1. **AI Hub - 한국형 자동차 번호판 이미지**
   - URL: https://aihub.or.kr/
   - 약 10만장 이상의 한국 번호판 데이터

2. **자체 수집**
   - 실제 환경에서 촬영한 번호판 이미지
   - 다양한 조명, 각도, 거리 조건

### 2. 레이블 형식

`labels.json` 파일 형식:

```json
{
  "plate_001.jpg": "12가3456",
  "plate_002.jpg": "301허1234",
  "plate_003.jpg": "서울12바3456",
  ...
}
```

### 3. 데이터 분할

- **학습 데이터**: 80% (약 8,000장 이상 권장)
- **검증 데이터**: 20% (약 2,000장 이상 권장)

## 데이터 증강

### 지원하는 증강 기법

1. **밝기 조정** - 다양한 조명 조건 시뮬레이션
2. **반사/글레어** - 유리 반사, 햇빛 반사
3. **야간 조건** - 저조도 환경
4. **흐림** - 모션 블러, 초점 흐림
5. **노이즈** - 가우시안, Salt & Pepper
6. **회전** - 각도 변화 (±15도)
7. **원근 변환** - 시점 변화

### 증강 프리셋

```python
from training.augmentation import PlateAugmentation, AUGMENTATION_PRESETS

augmenter = PlateAugmentation()

# 프리셋 사용
light_aug = augmenter.augment_pipeline(image, AUGMENTATION_PRESETS['light'])
heavy_aug = augmenter.augment_pipeline(image, AUGMENTATION_PRESETS['heavy'])

# 랜덤 증강
random_aug = augmenter.augment_random(image, num_augmentations=3)
```

## 모델 학습

### 기본 학습

```bash
python training/train_text_recognition.py \
  --train-dir data/train/images \
  --train-labels data/train/labels.json \
  --val-dir data/val/images \
  --val-labels data/val/labels.json \
  --epochs 50 \
  --batch-size 32 \
  --gpu \
  --augment
```

### 고급 옵션

```bash
python training/train_text_recognition.py \
  --train-dir data/train/images \
  --train-labels data/train/labels.json \
  --val-dir data/val/images \
  --val-labels data/val/labels.json \
  --epochs 100 \
  --batch-size 64 \
  --lr 0.0005 \
  --hidden-size 512 \
  --gpu \
  --augment \
  --num-workers 8 \
  --save-path models/crnn_korean_plate.pth
```

### 파라미터 설명

- `--train-dir`: 학습 이미지 디렉토리
- `--train-labels`: 학습 레이블 JSON 파일
- `--val-dir`: 검증 이미지 디렉토리
- `--val-labels`: 검증 레이블 JSON 파일
- `--epochs`: 학습 에폭 수 (기본: 50)
- `--batch-size`: 배치 크기 (기본: 32)
- `--lr`: 학습률 (기본: 0.001)
- `--hidden-size`: LSTM 은닉 크기 (기본: 256)
- `--gpu`: GPU 사용 플래그
- `--augment`: 데이터 증강 사용 플래그
- `--num-workers`: 데이터로더 워커 수 (기본: 4)
- `--save-path`: 모델 저장 경로

## 모델 아키텍처

### CRNN (Convolutional Recurrent Neural Network)

```
Input (1×32×128)
    ↓
CNN Feature Extractor
  - Conv1: 64 filters
  - Conv2: 128 filters
  - Conv3-4: 256 filters
  - Conv5-7: 512 filters
    ↓
Sequence Features (512×32)
    ↓
Bidirectional LSTM (2 layers, 256 hidden)
    ↓
Fully Connected Layer
    ↓
CTC Output (num_classes)
```

### CTC Loss

CTC (Connectionist Temporal Classification)를 사용하여 가변 길이 시퀀스 학습:
- 정렬 불필요
- 공백 문자 자동 처리
- End-to-end 학습

## 성능 평가

### 메트릭

1. **CER (Character Error Rate)**
   - 문자 단위 오류율
   - 낮을수록 좋음

2. **WER (Word Error Rate)**
   - 단어(번호판) 단위 오류율
   - 낮을수록 좋음

3. **정확도 (Accuracy)**
   - 완전 일치하는 번호판 비율
   - 높을수록 좋음

### 목표 성능

- **CER**: < 5%
- **WER**: < 10%
- **정확도**: > 90%

## 학습 로그 예시

```
Epoch 1/50
Training: 100%|████████| 250/250 [02:15<00:00, 1.85it/s, loss=1.234]
Train Loss: 1.2345
Validating: 100%|████████| 63/63 [00:15<00:00, 4.12it/s]
Val Loss: 0.8765, Val Accuracy: 0.7234
✓ 모델 저장 (정확도: 0.7234)

Epoch 2/50
Training: 100%|████████| 250/250 [02:14<00:00, 1.86it/s, loss=0.876]
Train Loss: 0.8765
Validating: 100%|████████| 63/63 [00:15<00:00, 4.15it/s]
Val Loss: 0.6543, Val Accuracy: 0.8123
✓ 모델 저장 (정확도: 0.8123)

...

Epoch 50/50
Training: 100%|████████| 250/250 [02:13<00:00, 1.87it/s, loss=0.123]
Train Loss: 0.1234
Validating: 100%|████████| 63/63 [00:15<00:00, 4.18it/s]
Val Loss: 0.0987, Val Accuracy: 0.9456
✓ 모델 저장 (정확도: 0.9456)

학습 완료! 최고 정확도: 0.9456
```

## 모델 사용

학습된 모델을 OCR 백엔드로 통합:

```python
# 1. 커스텀 백엔드 구현
from src.ocr.backends.base import OCRBackend
import torch

class CRNNBackend(OCRBackend):
    def __init__(self, model_path, charset):
        self.model = CRNN(len(charset) + 1)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.charset = charset

    # recognize, recognize_single 구현...

# 2. OCREngine에서 사용
# config.py: OCR_BACKEND = 'crnn'
```

## 문제 해결

### GPU 메모리 부족

배치 크기를 줄이세요:
```bash
--batch-size 16  # 또는 8
```

### 과적합

데이터 증강을 강화하거나 조기 종료를 사용하세요:
```bash
--augment  # 증강 활성화
```

### 학습이 안됨

학습률을 조정하세요:
```bash
--lr 0.0001  # 더 작은 학습률
```

## 참고 자료

1. Shi et al., "An End-to-End Trainable Neural Network for Image-based Sequence Recognition", TPAMI 2016
2. Graves et al., "Connectionist Temporal Classification", ICML 2006
3. AI Hub 한국형 자동차 번호판 이미지: https://aihub.or.kr/
