"""
슬롯 기반 문자 분류기 모듈

번호판 슬롯에서 개별 문자를 인식합니다.
docs/ocr_structured_pipeline_plan.md 기반 구현
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

# 허용 문자 정의
DIGITS = "0123456789"
HANGUL = ["사", "바", "배", "거", "노", "고", "서", "하"]  # 일반 번호판 한글
CLASSES = list(DIGITS) + HANGUL + ["_"]  # "_" = blank (빈 슬롯)


def extract_slots(image: np.ndarray, slots: List) -> List[np.ndarray]:
    """
    템플릿 메타데이터의 슬롯 정의에 따라 문자 영역 추출

    Args:
        image: 전처리된 번호판 이미지 (워프됨)
        slots: 슬롯 리스트 (TemplateMeta.slots)

    Returns:
        slot_images: 슬롯별 이미지 리스트
    """
    slot_images = []

    for slot in slots:
        # 슬롯 영역 잘라내기
        crop = image[slot.y:slot.y + slot.h, slot.x:slot.x + slot.w]

        # Grayscale 변환 (이미 그레이스케일이면 스킵)
        if len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 적응형 임계값 이진화
        # 배경은 검정, 글자는 흰색으로 만듬
        binary = cv2.adaptiveThreshold(
            crop, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # 글자를 흰색으로
            11, 2
        )

        slot_images.append(binary)

    return slot_images


class SlotClassifier:
    """
    슬롯 단위 문자 분류기

    EasyOCR 또는 커스텀 모델을 사용하여 슬롯별 문자를 인식합니다.
    """

    def __init__(self, model_path: Optional[str] = None, use_easyocr: bool = True):
        """
        Args:
            model_path: 커스텀 모델 경로 (.pth 또는 .pt)
            use_easyocr: EasyOCR 사용 여부 (True이면 모델 무시)
        """
        self.use_easyocr = use_easyocr
        self.model = None
        self.reader = None
        self.label_map = CLASSES

        if use_easyocr:
            # EasyOCR 리더 초기화
            try:
                import easyocr
                self.reader = easyocr.Reader(
                    ['ko', 'en'],
                    gpu=True,
                    verbose=False
                )
            except Exception as e:
                print(f"Warning: EasyOCR 초기화 실패 - {e}")
                self.reader = None

        else:
            # 커스텀 모델 로드
            if model_path and os.path.exists(model_path):
                try:
                    import torch
                    self.model = torch.jit.load(model_path)
                    self.model.eval()
                except Exception as e:
                    print(f"Warning: 커스텀 모델 로드 실패 - {e}")
                    self.model = None

    def predict(self, crops: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """
        슬롯 이미지들로부터 문자 예측

        Args:
            crops: 슬롯 이미지 리스트

        Returns:
            chars: 예측된 문자 리스트
            probs: 예측 확률 리스트 (0.0~1.0)
        """
        if self.use_easyocr and self.reader is not None:
            return self._predict_easyocr(crops)
        elif self.model is not None:
            return self._predict_custom(crops)
        else:
            # Fallback: 빈 결과 반환
            return ["_"] * len(crops), [0.0] * len(crops)

    def _predict_easyocr(self, crops: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """
        EasyOCR을 사용한 슬롯 문자 인식

        Args:
            crops: 슬롯 이미지 리스트

        Returns:
            chars: 예측된 문자 리스트
            probs: 예측 확률 리스트
        """
        chars = []
        probs = []

        # 허용 문자 리스트 (EasyOCR allowlist)
        allowlist = ''.join(DIGITS) + ''.join(HANGUL)

        for crop in crops:
            try:
                # EasyOCR 인식 (단일 슬롯)
                result = self.reader.readtext(
                    crop,
                    allowlist=allowlist,
                    detail=1,
                    paragraph=False
                )

                if result and len(result) > 0:
                    # 가장 신뢰도 높은 결과 선택
                    best_result = max(result, key=lambda x: x[2])
                    text = best_result[1]
                    confidence = best_result[2]

                    # 첫 글자만 사용 (슬롯은 1글자)
                    char = text[0] if text else "_"
                    chars.append(char)
                    probs.append(confidence)
                else:
                    # 인식 실패
                    chars.append("_")
                    probs.append(0.0)

            except Exception as e:
                # 에러 발생 시 빈 문자
                chars.append("_")
                probs.append(0.0)

        return chars, probs

    def _predict_custom(self, crops: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """
        커스텀 모델을 사용한 슬롯 문자 인식

        Args:
            crops: 슬롯 이미지 리스트

        Returns:
            chars: 예측된 문자 리스트
            probs: 예측 확률 리스트
        """
        import torch
        import torch.nn.functional as F

        chars = []
        probs = []

        # 이미지 전처리 및 배치 생성
        tensors = []
        for crop in crops:
            # 리사이즈 (예: 32x32)
            resized = cv2.resize(crop, (32, 32))
            # 정규화 (0~1)
            normalized = resized.astype(np.float32) / 255.0
            # 텐서 변환 (C, H, W)
            tensor = torch.from_numpy(normalized).unsqueeze(0)  # (1, H, W)
            tensors.append(tensor)

        # 배치 텐서 생성
        batch = torch.stack(tensors)  # (N, 1, H, W)

        # 추론
        with torch.no_grad():
            outputs = self.model(batch)  # (N, num_classes)
            probs_tensor = F.softmax(outputs, dim=1)  # (N, num_classes)
            confidences, indices = torch.max(probs_tensor, dim=1)

        # 결과 변환
        for idx, conf in zip(indices, confidences):
            char = self.label_map[idx.item()]
            chars.append(char)
            probs.append(conf.item())

        return chars, probs


def build_string(chars: List[str], slot_names: List[str]) -> str:
    """
    슬롯별 문자를 조합하여 번호판 문자열 생성

    Args:
        chars: 예측된 문자 리스트
        slot_names: 슬롯 이름 리스트

    Returns:
        text: 조합된 문자열
    """
    # "_" (blank) 제거
    filtered_chars = [c for c in chars if c != "_"]

    # 문자열 조합
    text = "".join(filtered_chars)

    return text


def recognize_plate_slots(processed_image: np.ndarray,
                          template_meta,
                          classifier: SlotClassifier) -> Tuple[str, List[str], List[float]]:
    """
    번호판 슬롯 기반 인식 (엔트리 포인트)

    Args:
        processed_image: 전처리된 번호판 이미지
        template_meta: 템플릿 메타데이터
        classifier: 슬롯 분류기 인스턴스

    Returns:
        text: 인식된 문자열
        chars: 슬롯별 문자 리스트
        probs: 슬롯별 확률 리스트
    """
    # 1. 슬롯 추출
    slot_crops = extract_slots(processed_image, template_meta.slots)

    # 2. 문자 분류
    chars, probs = classifier.predict(slot_crops)

    # 3. 문자열 조합
    slot_names = [s.name for s in template_meta.slots]
    text = build_string(chars, slot_names)

    return text, chars, probs
