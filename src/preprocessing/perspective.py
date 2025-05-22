import cv2
import numpy as np
import math

"""
원근 및 기울기 보정 모듈

이 모듈은 번호판 이미지의 기울기와 원근 왜곡을 보정하는 클래스를 제공합니다.
윤곽선 검출과 사각형 보정 알고리즘을 사용하여 번호판을 정면 시점으로 변환합니다.
"""
class PerspectiveCorrection:
    """원근 및 기울기 보정을 위한 클래스"""
    
    def __init__(self):
        """PerspectiveCorrection 클래스 초기화"""
        pass
    
    def correct(self, image):
        """
        이미지의 원근 및 기울기 보정
        
        Args:
            image (numpy.ndarray): 그레이스케일 또는 이진화된 이미지
            
        Returns:
            numpy.ndarray: 원근 및 기울기가 보정된 이미지
        """
        # 입력 이미지가 이진화되어 있지 않으면 이진화
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            if np.max(image) > 1:  # 이미 이진화된 경우
                binary = image.copy()
            else:
                _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
        # 윤곽선 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 윤곽선이 없으면 원본 이미지 반환
        if not contours:
            return image
        
        # 가장 큰 윤곽선 선택 (번호판 영역이라 가정)
        max_contour = max(contours, key=cv2.contourArea)
        
        # 윤곽선 면적이 너무 작으면 원본 이미지 반환
        if cv2.contourArea(max_contour) < 100:
            return image
        
        # 윤곽선을 둘러싸는 최소한의 회전된 직사각형 찾기
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # 회전 각도 계산
        angle = rect[2]
        
        # 각도가 -45°를 넘으면 90°에서 뺀다
        if angle < -45:
            angle = 90 + angle
            
        # 이미지 중심을 기준으로 회전 행렬 계산
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 회전 적용
        rotated = cv2.warpAffine(binary, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        
        # 회전 후 윤곽선 다시 검출
        contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 회전 후 윤곽선이 없으면 회전만 적용된 이미지 반환
        if not contours:
            return rotated
        
        # 가장 큰 윤곽선 선택
        max_contour = max(contours, key=cv2.contourArea)
        
        # 원근 보정을 위한 사각형 검출 (근사치 사용)
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        
        # 근사 윤곽선이 4점이 아니면 회전만 적용된 이미지 반환
        if len(approx) != 4:
            return rotated
        else:
            # 근사 윤곽선 4점 사용
            src_pts = approx.reshape(4, 2).astype(np.float32)
        
        # 사각형 정렬 (좌상, 우상, 우하, 좌하)
        src_pts = self.order_points(src_pts)
        
        # 목표 사각형 정의 (정면 시점)
        width = max(int(np.linalg.norm(src_pts[0] - src_pts[1])), 
                   int(np.linalg.norm(src_pts[2] - src_pts[3])))
        height = max(int(np.linalg.norm(src_pts[0] - src_pts[3])), 
                    int(np.linalg.norm(src_pts[1] - src_pts[2])))
        
        # width, height가 너무 작으면 회전만 적용된 이미지 반환
        if width < 10 or height < 10:
            return rotated
        
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # 원근 변환 행렬 계산
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 원근 변환 적용
        warped = cv2.warpPerspective(rotated, M, (width, height))
        
        return warped
    
    def order_points(self, pts):
        """
        사각형의 네 점을 좌상, 우상, 우하, 좌하 순서로 정렬
        
        Args:
            pts (numpy.ndarray): 정렬할 4개의 점 좌표
            
        Returns:
            numpy.ndarray: 정렬된 점 좌표
        """
        # x+y 값을 기준으로 정렬하면 좌상(최소), 우하(최대)를 찾을 수 있음
        s = pts.sum(axis=1)
        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = pts[np.argmin(s)]  # 좌상
        rect[2] = pts[np.argmax(s)]  # 우하
        
        # 우상, 좌하를 구분하기 위해 x-y 값 사용
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 우상
        rect[3] = pts[np.argmax(diff)]  # 좌하
        
        return rect
    
    def detect_rotation_angle(self, image):
        """
        이미지의 회전 각도 감지
        
        Args:
            image (numpy.ndarray): 그레이스케일 또는 이진화된 이미지
            
        Returns:
            float: 감지된 회전 각도(도)
        """
        # 이미지가 그레이스케일이 아니면 변환
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 모든 윤곽선에 대한 각도 계산
        angles = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:  # 너무 작은 윤곽선 무시
                continue
                
            # 윤곽선을 둘러싸는 최소한의 회전된 직사각형 찾기
            rect = cv2.minAreaRect(contour)
            
            # 각도 조정 (-90° ~ 0°)
            angle = rect[2]
            if angle < -45:
                angle = 90 + angle
                
            angles.append(angle)
        
        # 각도가 없으면 0 반환
        if not angles:
            return 0
            
        # 중간값 각도 반환 (이상치에 강인)
        return np.median(angles)