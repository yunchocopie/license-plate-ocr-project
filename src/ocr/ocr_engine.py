import cv2
import numpy as np
import easyocr
import re
import torch
from .text_postprocess import TextPostProcessor
import config

"""
OCR 엔진 모듈

이 모듈은 번호판 이미지의 문자를 인식하는 OCR 엔진 클래스를 제공합니다.
EasyOCR 라이브러리를 사용하여 한국어와 영어 문자를 인식합니다.
"""
class OCREngine:
    """번호판 문자 인식을 위한 OCR 엔진 클래스"""
    
    def __init__(self, languages=None, gpu=None, allowed_chars=None):
        """
        OCREngine 클래스 초기화
        
        Args:
            languages (list, optional): 인식할 언어 목록. 기본값은 config에서 가져옴
            gpu (bool, optional): GPU 사용 여부. 기본값은 config에서 가져옴
            allowed_chars (str, optional): 허용된 문자 집합. 기본값은 config에서 가져옴
        """
        self.languages = languages or config.OCR_LANGUAGES
        self.gpu = gpu if gpu is not None else config.OCR_GPU
        self.allowed_chars = allowed_chars or config.OCR_ALLOWED_CHARS
        
        # GPU 사용 가능 여부 확인
        if self.gpu and not torch.cuda.is_available():
            self.gpu = False
            print("GPU not available, using CPU instead.")
        
        # EasyOCR 리더 초기화
        self.reader = easyocr.Reader(
            self.languages,
            gpu=self.gpu,
            model_storage_directory=config.MODEL_DIR,
            download_enabled=True
        )
        
        # 텍스트 후처리기 초기화
        self.post_processor = TextPostProcessor()
    
    def recognize(self, image, detail=0):
        """
        이미지에서 텍스트 인식
        
        Args:
            image (numpy.ndarray): 그레이스케일 또는 이진화된 이미지
            detail (int, optional): 결과 상세 레벨. 기본값은 0
            
        Returns:
            str: 인식된 번호판 텍스트
        """
        # 이미지가 None이거나 비어있으면 빈 문자열 반환
        if image is None or image.size == 0:
            return ""
        
        # 이미지 전처리 (이미지가 이미 그레이스케일인지 확인)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # OCR 수행
        try:
            results = self.reader.readtext(gray, detail=detail, allowlist=self.allowed_chars)
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
        
        # 결과가 없으면 빈 문자열 반환
        if not results:
            return ""
        
        # 상세 레벨에 따라 결과 처리
        if detail == 0:  # 텍스트만 반환
            if isinstance(results, list):
                # 결과가 리스트인 경우 공백으로 연결
                text = " ".join(results)
            else:
                text = str(results)
        else:  # 바운딩 박스, 신뢰도 등 포함 반환
            # 신뢰도 기준으로 결과 정렬
            results.sort(key=lambda x: x[2], reverse=True)
            
            # 텍스트 추출 (신뢰도 내림차순)
            text = " ".join([result[1] for result in results])
        
        # 텍스트 후처리
        processed_text = self.post_processor.process(text)
        
        return processed_text
    
    def recognize_with_confidence(self, image, min_confidence=0.3):
        """
        신뢰도 정보를 포함하여 텍스트 인식
        
        Args:
            image (numpy.ndarray): 그레이스케일 또는 이진화된 이미지
            min_confidence (float, optional): 최소 신뢰도 임계값. 기본값은 0.3
            
        Returns:
            tuple: (인식된 텍스트, 신뢰도)
        """
        # 이미지가 None이거나 비어있으면 빈 결과 반환
        if image is None or image.size == 0:
            return "", 0.0
        
        # 이미지 전처리
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # OCR 수행 (상세 정보 포함)
        try:
            results = self.reader.readtext(gray, detail=1, allowlist=self.allowed_chars)
        except Exception as e:
            print(f"OCR Error: {e}")
            return "", 0.0
        
        # 결과가 없으면 빈 결과 반환
        if not results:
            return "", 0.0
        
        # 신뢰도를 기준으로 결과 필터링 및 정렬
        filtered_results = [r for r in results if r[2] >= min_confidence]
        filtered_results.sort(key=lambda x: x[2], reverse=True)
        
        if not filtered_results:
            return "", 0.0
        
        # 인식된 텍스트와 평균 신뢰도 계산
        texts = [r[1] for r in filtered_results]
        confidences = [r[2] for r in filtered_results]
        
        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences)
        
        # 텍스트 후처리
        processed_text = self.post_processor.process(combined_text)
        
        return processed_text, avg_confidence
    
    def recognize_korean_license_plate(self, image):
        """
        한국 차량 번호판 전용 인식 (번호판 형식에 맞게 처리)
        
        Args:
            image (numpy.ndarray): 그레이스케일 또는 이진화된 이미지
            
        Returns:
            str: 한국 번호판 형식에 맞게 처리된 텍스트
        """
        # 기본 OCR 인식
        text, confidence = self.recognize_with_confidence(image)
        
        # 한국 번호판 형식에 맞게 후처리
        plate_text = self.post_processor.format_korean_license_plate(text)
        
        return plate_text
    
    def test_preprocess_variations(self, image):
        """
        다양한 전처리 방법을 시도하여 최상의 OCR 결과 찾기
        
        Args:
            image (numpy.ndarray): 원본 이미지
            
        Returns:
            dict: 전처리 방법별 결과와 신뢰도
        """
        results = {}
        
        # 1. 원본 이미지
        original_text, original_conf = self.recognize_with_confidence(image)
        results["original"] = (original_text, original_conf)
        
        # 2. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_text, gray_conf = self.recognize_with_confidence(gray)
            results["grayscale"] = (gray_text, gray_conf)
        
        # 3. 이진화 (OTSU)
        if len(image.shape) == 3:
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:
            binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
        binary_text, binary_conf = self.recognize_with_confidence(binary)
        results["binary_otsu"] = (binary_text, binary_conf)
        
        # 4. 적응형 이진화
        if len(image.shape) == 3:
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
        adaptive_text, adaptive_conf = self.recognize_with_confidence(adaptive)
        results["binary_adaptive"] = (adaptive_text, adaptive_conf)
        
        # 5. 대비 향상
        if len(image.shape) == 3:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
        enhanced_text, enhanced_conf = self.recognize_with_confidence(enhanced)
        results["enhanced"] = (enhanced_text, enhanced_conf)
        
        # 6. 노이즈 제거
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
        denoised_text, denoised_conf = self.recognize_with_confidence(denoised)
        results["denoised"] = (denoised_text, denoised_conf)
        
        # 최적의 결과 찾기 (가장 높은 신뢰도)
        best_method = max(results.items(), key=lambda x: x[1][1])
        results["best"] = best_method[1]
        results["best_method"] = best_method[0]
        
        return results