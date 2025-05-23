import cv2
import numpy as np
from ultralytics import YOLO
import torch
import config

"""
번호판 검출 모듈

이 모듈은 차량 이미지에서 번호판을 검출하는 클래스를 제공합니다.
license-plate-ocr-project의 코드를 기반으로 번호판을 감지합니다.
"""
class PlateDetector:
    """번호판 검출을 위한 클래스"""
    
    def __init__(self, model_path=None, conf_threshold=None):
        """
        PlateDetector 클래스 초기화
        
        Args:
            model_path (str, optional): YOLO 모델 경로. 기본값은 config에서 가져옴
            conf_threshold (float, optional): 신뢰도 임계값. 기본값은 config에서 가져옴
        """
        self.model_path = model_path or config.PLATE_DETECTION_MODEL
        self.conf_threshold = conf_threshold or config.PLATE_DETECTION_CONF
        
        # YOLO 모델 로드
        self.model = YOLO(self.model_path)
        
        # GPU 사용 가능 시 GPU 사용
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Plate Detector using device: {self.device}")
    
    def detect(self, image):
        """
        이미지에서 번호판 검출 (license-plate-ocr-project 로직 통합)
        
        Args:
            image (numpy.ndarray): BGR 형식의 차량 이미지
            
        Returns:
            list: 검출된 번호판의 바운딩 박스 목록 [x1, y1, x2, y2]
        """
        # YOLO 모델로 번호판 감지 (이미지 크기 조정은 YOLO 내부에서 처리)
        results = self.model(image, conf=self.conf_threshold, device=self.device)
        
        boxes = []
        for r in results:
            # 모든 감지된 바운딩 박스
            for box in r.boxes:
                # 클래스 확인 (plate 클래스만 선택)
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                if class_name != 'plate':  # 번호판 클래스가 아니면 무시
                    continue
                
                # 바운딩 박스 좌표 추출
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
        
        return boxes
    
    def detect_best_plate(self, image):
        """
        이미지에서 신뢰도가 가장 높은 번호판 하나만 검출
        
        Args:
            image (numpy.ndarray): BGR 형식의 차량 이미지
            
        Returns:
            numpy.ndarray: 크롭된 번호판 이미지 또는 None
        """
        # YOLO 모델로 번호판 감지
        results = self.model(image, conf=self.conf_threshold, device=self.device)
        
        for r in results:
            boxes = r.boxes  # 감지된 객체들의 바운딩 박스들
            best_plate = None  # 가장 높은 확신도의 박스를 저장하기 위한 변수
            best_conf = -1  # 가장 높은 신뢰도 저장
            
            for box in boxes:  # 가장 높은 신뢰도의 번호판을 찾기 위한 반복문
                cls_id = int(box.cls[0])  # 클래스 ID
                class_name = self.model.names[cls_id]  # 클래스 이름
                
                if class_name != 'plate':  # 번호판이 아닌 클래스는 무시
                    continue
                
                conf = box.conf[0].item()  # 현재 박스의 신뢰도
                if conf > best_conf:  # 가장 높은 신뢰도의 박스를 찾기
                    best_conf = conf
                    best_plate = box
            
            if best_plate:  # 가장 신뢰도가 높은 번호판이 있다면 크롭
                x1, y1, x2, y2 = map(int, best_plate.xyxy[0].tolist())
                cropped = image[y1:y2, x1:x2]
                return cropped
                
        return None
    
    def visualize(self, image, boxes):
        """
        이미지에 검출된 번호판의 바운딩 박스 시각화
        
        Args:
            image (numpy.ndarray): BGR 형식의 이미지
            boxes (list): 바운딩 박스 목록 [x1, y1, x2, y2]
            
        Returns:
            numpy.ndarray: 바운딩 박스가 그려진 이미지
        """
        vis_img = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_img, "License Plate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return vis_img
    
    def crop_plates(self, image, boxes, padding=5):
        """
        이미지에서 번호판 영역 추출
        
        Args:
            image (numpy.ndarray): BGR 형식의 이미지
            boxes (list): 바운딩 박스 목록 [x1, y1, x2, y2]
            padding (int, optional): 바운딩 박스 주변 여백. 기본값은 5
            
        Returns:
            list: 추출된 번호판 이미지 목록
        """
        h, w = image.shape[:2]
        plate_images = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # 패딩 추가 (이미지 경계 초과 방지)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 번호판 영역 추출
            plate_img = image[y1:y2, x1:x2]
            
            if plate_img.size > 0:  # 유효한 이미지인지 확인
                plate_images.append(plate_img)
        
        return plate_images