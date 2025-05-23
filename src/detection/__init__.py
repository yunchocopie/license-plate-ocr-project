"""
차량 및 번호판 검출 관련 모듈

이 패키지는 차량과 번호판을 검출하는 클래스와 함수들을 포함합니다.
"""

from .vehicle_detector import VehicleDetector
from .plate_detector import PlateDetector

__all__ = ['VehicleDetector', 'PlateDetector']
