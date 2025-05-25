import cv2
import numpy as np
import easyocr
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
        self.download_enabled = download_enabled if download_enabled is not None else config.DOWNLOAD_ENABLED

        self.reader = easyocr.Reader(
            self.languages,
            gpu=self.gpu,
            model_storage_directory=self.model_storage_directory,
            download_enabled=self.download_enabled
        )
        self.post_processor = TextPostProcessor(allowed_chars=self.allowed_chars)

    def recognize_with_confidence(self, image, min_confidence=None):
        min_confidence = min_confidence if min_confidence is not None else config.MIN_OCR_CONFIDENCE
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
