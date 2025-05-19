import cv2
import numpy as np
from .blur_correction import BlurCorrection
from .perspective import PerspectiveCorrection
from .normalize import Normalize
import config

"""
이미지 기본 처리 모듈

이 모듈은 번호판 이미지 전처리의 주요 클래스를 제공합니다.
흐림 보정, 기울기 보정, 비율 정규화 등의 작업을 순차적으로 적용합니다.
"""
class ImageProcessor:
    """번호판 이미지 전처리를 위한 클래스"""
    
    def __init__(self):
        """ImageProcessor 클래스 초기화"""
        # 각 전처리 모듈 초기화
        self.blur_corrector = BlurCorrection()
        self.perspective_corrector = PerspectiveCorrection()
        self.normalizer = Normalize()
        
    def process(self, image):
        """
        번호판 이미지 전처리 실행
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            
        Returns:
            numpy.ndarray: 전처리된 번호판 이미지
        """
        # 이미지 복사본 생성
        processed_img = image.copy()
        
        # 그레이스케일 변환
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_img.copy()
            
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 흐림 보정
        deblurred = self.blur_corrector.correct(denoised)
        
        # 이진화 (번호판은 밝은 배경에 어두운 문자)
        _, binary = cv2.threshold(deblurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 기울기/원근 보정
        warped = self.perspective_corrector.correct(binary)
        
        # 비율 정규화
        normalized = self.normalizer.normalize(warped)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # 추가적인 노이즈 제거
        final = cv2.medianBlur(enhanced, 3)
        
        return final
    
    def apply_individual(self, image, blur=True, perspective=True, normalize=True, enhance=True):
        """
        개별 전처리 단계 선택적 적용
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            blur (bool, optional): 흐림 보정 적용 여부. 기본값은 True
            perspective (bool, optional): 기울기 보정 적용 여부. 기본값은 True
            normalize (bool, optional): 비율 정규화 적용 여부. 기본값은 True
            enhance (bool, optional): 대비 향상 적용 여부. 기본값은 True
            
        Returns:
            numpy.ndarray: 선택된 전처리 단계가 적용된 이미지
        """
        # 이미지 복사본 생성
        processed_img = image.copy()
        
        # 그레이스케일 변환
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_img.copy()
            
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        current = denoised
        
        # 흐림 보정
        if blur:
            current = self.blur_corrector.correct(current)
            
        # 이진화
        _, binary = cv2.threshold(current, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        current = binary
        
        # 기울기/원근 보정
        if perspective:
            current = self.perspective_corrector.correct(current)
            
        # 비율 정규화
        if normalize:
            current = self.normalizer.normalize(current)
            
        # 대비 향상
        if enhance:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            current = clahe.apply(current)
            
        return current
    
    def visualize_steps(self, image):
        """
        전처리 단계별 시각화
        
        Args:
            image (numpy.ndarray): BGR 형식의 번호판 이미지
            
        Returns:
            dict: 각 전처리 단계별 이미지 딕셔너리
        """
        # 이미지 복사본 생성
        original = image.copy()
        
        # 그레이스케일 변환
        if len(original.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original.copy()
            
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 흐림 보정
        deblurred = self.blur_corrector.correct(denoised)
        
        # 이진화
        _, binary = cv2.threshold(deblurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 기울기/원근 보정
        warped = self.perspective_corrector.correct(binary)
        
        # 비율 정규화
        normalized = self.normalizer.normalize(warped)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # 단계별 이미지 딕셔너리 반환
        return {
            'original': original,
            'gray': gray,
            'denoised': denoised,
            'deblurred': deblurred,
            'binary': binary,
            'warped': warped,
            'normalized': normalized,
            'enhanced': enhanced,
            'final': cv2.medianBlur(enhanced, 3)
        }