"""
OCR 백엔드 추상 클래스

모든 OCR 백엔드가 구현해야 하는 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import numpy as np


class OCRBackend(ABC):
    """OCR 백엔드 추상 클래스"""

    @abstractmethod
    def __init__(self, languages: List[str], gpu: bool = False, **kwargs):
        """
        OCR 백엔드 초기화

        Args:
            languages: 인식할 언어 목록
            gpu: GPU 사용 여부
            **kwargs: 백엔드별 추가 파라미터
        """
        pass

    @abstractmethod
    def recognize(self, image: np.ndarray, **kwargs) -> List[Tuple[List, str, float]]:
        """
        이미지에서 텍스트 인식

        Args:
            image: 입력 이미지 (numpy array)
            **kwargs: 백엔드별 추가 파라미터

        Returns:
            List[Tuple[List, str, float]]: (바운딩 박스, 텍스트, 신뢰도) 리스트
        """
        pass

    @abstractmethod
    def recognize_single(self, image: np.ndarray, **kwargs) -> Tuple[str, float]:
        """
        단일 텍스트 영역 인식

        Args:
            image: 입력 이미지 (numpy array)
            **kwargs: 백엔드별 추가 파라미터

        Returns:
            Tuple[str, float]: (인식된 텍스트, 신뢰도)
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        지원하는 언어 목록 반환

        Returns:
            List[str]: 지원 언어 코드 리스트
        """
        pass

    @abstractmethod
    def is_gpu_available(self) -> bool:
        """
        GPU 사용 가능 여부 확인

        Returns:
            bool: GPU 사용 가능 여부
        """
        pass

    def get_backend_info(self) -> Dict[str, Any]:
        """
        백엔드 정보 반환

        Returns:
            Dict: 백엔드 이름, 버전 등 정보
        """
        return {
            'name': self.__class__.__name__,
            'gpu_available': self.is_gpu_available(),
            'supported_languages': self.get_supported_languages()
        }
