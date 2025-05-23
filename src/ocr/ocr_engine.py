
import cv2
import numpy as np
import easyocr
import re
import torch
from .text_postprocess import TextPostProcessor # 상대 경로 유지
import config # config 파일 임포트

class OCREngine:
    def __init__(self, languages=None, gpu=None, allowed_chars=None, model_storage_directory=None, download_enabled=None):
        self.languages = languages if languages is not None else config.OCR_LANGUAGES
        self.gpu = gpu if gpu is not None else config.OCR_GPU

        if self.gpu and not torch.cuda.is_available():
            print("WARNING: GPU not available or PyTorch not compiled with CUDA support, using CPU instead.")
            self.gpu = False
        elif self.gpu and torch.cuda.is_available():
            print("INFO: GPU available, using GPU for OCR.")
        else:
            print("INFO: Using CPU for OCR.")

        # EasyOCR은 문자열 리스트를 허용 문자로 받음
        self.allowed_chars = allowed_chars if allowed_chars is not None else config.OCR_ALLOWED_CHARS
        self.model_storage_directory = model_storage_directory if model_storage_directory is not None else config.MODEL_DIR
        self.download_enabled = download_enabled if download_enabled is not None else True # 기본값 True로 설정

        self.reader = easyocr.Reader(
            self.languages,
            gpu=self.gpu,
            model_storage_directory=self.model_storage_directory,
            download_enabled=self.download_enabled
        )
        self.post_processor = TextPostProcessor(allowed_chars=self.allowed_chars)


    def recognize(self, image, detail=0): # 입력은 ImageProcessor에서 처리된 그레이스케일 이미지로 가정
        if image is None or image.size == 0:
            return ""

        # 입력 이미지가 uint8 타입인지 확인, 아니면 변환
        if image.dtype != np.uint8:
            print(f"OCREngine: Input image dtype is {image.dtype}, converting to uint8.")
            if np.max(image) <= 1.0 and (image.dtype == np.float32 or image.dtype == np.float64) :
                processed_image = (image * 255).astype(np.uint8)
            else:
                processed_image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            processed_image = image

        # EasyOCR의 readtext 파라미터 추가 가능 (예: paragraph=False)
        try:
            results = self.reader.readtext(processed_image, detail=detail, allowlist=self.allowed_chars, paragraph=False)
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

        if not results:
            return ""

        # 결과 처리 로직은 기존과 유사하게 유지
        if detail == 0:
            if isinstance(results, list) and all(isinstance(item, str) for item in results):
                text = " ".join(results)
            elif isinstance(results, list) and all(isinstance(item, tuple) and len(item) > 1 for item in results): # [(bbox, text, conf), ...]
                 text = " ".join([res[1] for res in results])
            elif isinstance(results, str):
                 text = results
            else: # 예상치 못한 결과 형태
                 print(f"Unexpected OCR result format: {results}")
                 text = ""
        else: # detail=1
            # 신뢰도 기준으로 정렬 (높은 것이 먼저 오도록)
            # EasyOCR 결과는 이미 정렬되어 있을 수 있으나, 명시적으로 정렬
            # results.sort(key=lambda x: x[0][0][0]) # X 좌표 기준으로 정렬 (읽는 순서)
            text = " ".join([result[1] for result in results])

        processed_text = self.post_processor.process(text)
        return processed_text

    def recognize_with_confidence(self, image, min_confidence=0.3):
        # (기존 로직과 유사하게, recognize 메서드와 입력 이미지 처리 동일하게)
        if image is None or image.size == 0:
            return "", 0.0

        if image.dtype != np.uint8:
            if np.max(image) <= 1.0 and (image.dtype == np.float32 or image.dtype == np.float64) :
                processed_image = (image * 255).astype(np.uint8)
            else:
                processed_image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            processed_image = image

        try:
            results = self.reader.readtext(processed_image, detail=1, allowlist=self.allowed_chars, paragraph=False)
        except Exception as e:
            print(f"OCR Error: {e}")
            return "", 0.0

        if not results:
            return "", 0.0

        # 신뢰도 필터링 및 정렬
        filtered_results = [r for r in results if r[2] >= min_confidence]
        # filtered_results.sort(key=lambda x: x[0][0][0]) # X 좌표 기준 정렬

        if not filtered_results:
            return "", 0.0

        texts = [r[1] for r in filtered_results]
        confidences = [r[2] for r in filtered_results]

        combined_text = "".join(texts) # 번호판은 공백 없이 합치는 것이 나을 수 있음
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        processed_text = self.post_processor.process(combined_text)
        return processed_text, avg_confidence

    def recognize_korean_license_plate(self, image):
        # 이 함수는 recognize_with_confidence를 사용하므로 별도 수정은 적음
        # 다만, TextPostProcessor의 format_korean_license_plate가 중요
        text, confidence = self.recognize_with_confidence(image)
        plate_text = self.post_processor.format_korean_license_plate(text)
        return plate_text

    # test_preprocess_variations 메서드는 디버깅에 유용하므로 유지하고,
    # ImageProcessor의 다양한 옵션을 테스트하도록 확장 가능
