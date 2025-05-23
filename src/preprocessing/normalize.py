import cv2
import numpy as np
import config

"""
비율 정규화 모듈

이 모듈은 번호판 이미지의 크기와 비율을 표준화하는 클래스를 제공합니다.
OCR 엔진의 인식률을 높이기 위해 번호판 이미지의 비율을 통일합니다.
"""
class Normalize:
    """번호판 이미지 비율 정규화를 위한 클래스"""
    
    def __init__(self, target_size=None):
        """
        Normalize 클래스 초기화
        
        Args:
            target_size (tuple, optional): 목표 이미지 크기 (너비, 높이). 기본값은 config에서 가져옴
        """
        self.target_size = target_size or config.PLATE_SIZE
    
    def normalize(self, image):
        """
        이미지 비율 정규화
        
        Args:
            image (numpy.ndarray): 입력 이미지
            
        Returns:
            numpy.ndarray: 비율이 정규화된 이미지
        """
        # 입력 이미지가 비어 있거나 손상된 경우 빈 이미지 반환
        if image is None or image.size == 0:
            return np.zeros(self.target_size, dtype=np.uint8)
        
        # 이진화된 이미지인지 확인하고 그레이스케일로 통일
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 현재 이미지 크기
        h, w = gray.shape[:2]
        
        # 목표 크기
        target_w, target_h = self.target_size
        
        # 현재 비율 계산
        current_ratio = w / h
        
        # 목표 비율 계산
        target_ratio = target_w / target_h
        
        # 비율에 따라 크기 조정 방법 결정
        if current_ratio > target_ratio:
            # 너비 기준 조정
            new_w = target_w
            new_h = int(new_w / current_ratio)
            # 상하 패딩 추가
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_left, pad_right = 0, 0
        else:
            # 높이 기준 조정
            new_h = target_h
            new_w = int(new_h * current_ratio)
            # 좌우 패딩 추가
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            pad_top, pad_bottom = 0, 0
        
        # 이미지 크기 조정
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 패딩 추가
        normalized = cv2.copyMakeBorder(
            resized, 
            pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, 
            value=255  # 흰색 패딩 (번호판 배경색)
        )
        
        return normalized
    
    def adaptive_normalize(self, image):
        """
        적응형 비율 정규화 (텍스트 컨텐츠에 맞게 조정)
        
        Args:
            image (numpy.ndarray): 입력 이미지
            
        Returns:
            numpy.ndarray: 비율이 정규화된 이미지
        """
        # 입력 이미지가 비어 있거나 손상된 경우 빈 이미지 반환
        if image is None or image.size == 0:
            return np.zeros(self.target_size, dtype=np.uint8)
        
        # 이진화된 이미지인지 확인하고 그레이스케일로 통일
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 텍스트 컨텐츠 영역 찾기
        contours, _ = cv2.findContours(cv2.bitwise_not(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 컨텐츠가 없으면 기본 정규화 사용
            return self.normalize(gray)
        
        # 모든 컨텐츠 영역을 포함하는 바운딩 박스 찾기
        x_min, y_min = np.inf, np.inf
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # 유효한 컨텐츠 영역이 없으면 기본 정규화 사용
        if x_min >= x_max or y_min >= y_max:
            return self.normalize(gray)
        
        # 컨텐츠 주변에 여백 추가 (전체 이미지의 10%)
        h, w = gray.shape[:2]
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        
        # 컨텐츠 영역 추출
        content = gray[y_min:y_max, x_min:x_max]
        
        # 추출된 컨텐츠 정규화
        return self.normalize(content)
    
    def remove_border(self, image, border_thickness=5):
        """
        번호판 테두리 제거
        
        Args:
            image (numpy.ndarray): 입력 이미지
            border_thickness (int, optional): 제거할 테두리 두께. 기본값은 5
            
        Returns:
            numpy.ndarray: 테두리가 제거된 이미지
        """
        # 입력 이미지가 비어 있거나 손상된 경우 빈 이미지 반환
        if image is None or image.size == 0:
            return np.zeros(self.target_size, dtype=np.uint8)
        
        # 이미지 크기 확인
        h, w = image.shape[:2]
        
        # 테두리 제거 (단순히 이미지 가장자리를 잘라냄)
        if h > 2 * border_thickness and w > 2 * border_thickness:
            cropped = image[border_thickness:h-border_thickness, border_thickness:w-border_thickness]
            return cropped
        else:
            return image