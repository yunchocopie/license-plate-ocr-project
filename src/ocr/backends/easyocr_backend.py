"""
EasyOCR 백엔드 구현

EasyOCR 엔진을 OCRBackend 인터페이스로 래핑
"""

import easyocr
import torch
import os
from typing import List, Tuple, Dict, Any
import numpy as np
from .base import OCRBackend
from ...utils.logger import setup_logger
import config

logger = setup_logger(__name__)


class EasyOCRBackend(OCRBackend):
    """EasyOCR 백엔드 구현"""

    def __init__(self, languages: List[str] = None, gpu: bool = None,
                 model_storage_directory: str = None, download_enabled: bool = False,
                 allowed_chars: str = None, **kwargs):
        """
        EasyOCR 백엔드 초기화

        Args:
            languages: 인식할 언어 목록 (기본값: config.OCR_LANGUAGES)
            gpu: GPU 사용 여부 (기본값: config.OCR_GPU)
            model_storage_directory: 모델 저장 디렉토리
            download_enabled: 모델 다운로드 허용 여부
            allowed_chars: 허용할 문자 집합
            **kwargs: 추가 파라미터
        """
        self.languages = languages if languages is not None else config.OCR_LANGUAGES
        self.gpu = gpu if gpu is not None else config.OCR_GPU
        self.model_storage_directory = model_storage_directory or config.MODEL_DIR
        self.download_enabled = download_enabled
        self.allowed_chars = allowed_chars or config.OCR_ALLOWED_CHARS

        # GPU 사용 가능 여부 확인
        if self.gpu and not torch.cuda.is_available():
            logger.warning("GPU not available or PyTorch not compiled with CUDA support, using CPU instead.")
            self.gpu = False
        elif self.gpu and torch.cuda.is_available():
            logger.info("EasyOCR: GPU available, using GPU for OCR.")
        else:
            logger.info("EasyOCR: Using CPU for OCR.")

        # 로컬 모델 파일 경로 설정 및 확인
        self.craft_model_path = os.path.join(self.model_storage_directory, 'craft_mlt_25k.pth')
        self.korean_model_path = os.path.join(self.model_storage_directory, 'korean_g2.pth')

        # 로컬 모델 파일 존재 여부 확인
        if not os.path.exists(self.craft_model_path):
            logger.warning(f"CRAFT 모델 파일을 찾을 수 없습니다: {self.craft_model_path}")
            if not self.download_enabled:
                raise FileNotFoundError(f"CRAFT 모델 파일을 찾을 수 없습니다: {self.craft_model_path}")

        if not os.path.exists(self.korean_model_path):
            logger.warning(f"한국어 모델 파일을 찾을 수 없습니다: {self.korean_model_path}")
            if not self.download_enabled:
                raise FileNotFoundError(f"한국어 모델 파일을 찾을 수 없습니다: {self.korean_model_path}")

        logger.info(f"로컬 모델 사용: CRAFT={self.craft_model_path}, Korean={self.korean_model_path}")

        # EasyOCR Reader 초기화
        try:
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=self.download_enabled,
                verbose=True
            )
            logger.info("EasyOCR Reader 초기화 완료")
        except Exception as e:
            logger.error(f"EasyOCR Reader 초기화 실패: {e}")
            raise

    def recognize(self, image: np.ndarray, detail: int = 1, paragraph: bool = False,
                  min_size: int = 10, **kwargs) -> List[Tuple[List, str, float]]:
        """
        이미지에서 텍스트 인식

        Args:
            image: 입력 이미지 (numpy array)
            detail: 상세 레벨 (0: 텍스트만, 1: 바운딩박스+텍스트+신뢰도)
            paragraph: 단락 모드
            min_size: 최소 텍스트 크기
            **kwargs: EasyOCR 추가 파라미터

        Returns:
            List[Tuple[List, str, float]]: (바운딩 박스, 텍스트, 신뢰도) 리스트
        """
        try:
            # allowlist 설정
            if self.allowed_chars:
                kwargs['allowlist'] = self.allowed_chars

            results = self.reader.readtext(
                image,
                detail=detail,
                paragraph=paragraph,
                min_size=min_size,
                **kwargs
            )
            return results
        except Exception as e:
            logger.error(f"EasyOCR 인식 실패: {e}")
            return []

    def recognize_single(self, image: np.ndarray, **kwargs) -> Tuple[str, float]:
        """
        단일 텍스트 영역 인식

        Args:
            image: 입력 이미지 (numpy array)
            **kwargs: EasyOCR 추가 파라미터

        Returns:
            Tuple[str, float]: (인식된 텍스트, 신뢰도)
        """
        results = self.recognize(image, **kwargs)

        if not results:
            return "", 0.0

        # 가장 신뢰도 높은 결과 선택
        best_result = max(results, key=lambda x: x[2] if len(x) > 2 else 0)

        if len(best_result) >= 3:
            text = best_result[1]
            confidence = best_result[2]
            return text, confidence
        elif len(best_result) >= 2:
            text = best_result[1]
            return text, 0.0
        else:
            return "", 0.0

    def get_supported_languages(self) -> List[str]:
        """
        지원하는 언어 목록 반환

        Returns:
            List[str]: 지원 언어 코드 리스트
        """
        # EasyOCR 지원 언어 목록 (주요 언어만)
        return [
            'ko', 'en', 'ch_sim', 'ch_tra', 'ja', 'th', 'vi',
            'ar', 'ru', 'de', 'fr', 'es', 'pt', 'it', 'nl',
            'pl', 'tr', 'fa', 'hi', 'bn', 'ta', 'te'
        ]

    def is_gpu_available(self) -> bool:
        """
        GPU 사용 가능 여부 확인

        Returns:
            bool: GPU 사용 가능 여부
        """
        return self.gpu and torch.cuda.is_available()

    def get_backend_info(self) -> Dict[str, Any]:
        """
        백엔드 정보 반환

        Returns:
            Dict: 백엔드 이름, 버전 등 정보
        """
        info = super().get_backend_info()
        info.update({
            'backend': 'EasyOCR',
            'version': easyocr.__version__ if hasattr(easyocr, '__version__') else 'unknown',
            'languages': self.languages,
            'model_directory': self.model_storage_directory,
            'allowed_chars_count': len(self.allowed_chars) if self.allowed_chars else 0
        })
        return info

    def set_allowed_chars(self, allowed_chars: str):
        """
        허용 문자 집합 설정

        Args:
            allowed_chars: 허용할 문자 문자열
        """
        self.allowed_chars = allowed_chars
        logger.info(f"허용 문자 업데이트: {len(allowed_chars)}개 문자")
