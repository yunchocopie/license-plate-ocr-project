"""
OCR 백엔드 모듈

다양한 OCR 엔진을 통합하기 위한 백엔드 인터페이스
"""

from .base import OCRBackend
from .easyocr_backend import EasyOCRBackend

__all__ = ['OCRBackend', 'EasyOCRBackend']
