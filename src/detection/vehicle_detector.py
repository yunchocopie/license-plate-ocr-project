import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
import config

"""
차량 검출 모듈

이 모듈은 이미지 또는 비디오 프레임에서 차량을 검출하는 클래스를 제공합니다.
YOLOv8s 모델을 사용하여 차량을 감지합니다.
"""
class VehicleDetector:
    """차량 검출을 위한 클래스"""
    
    def __init__(self, model_path=None, conf_threshold=None):
        """
        VehicleDetector 클래스 초기화
        
        Args:
            model_path (str, optional): YOLO 모델 경로. 기본값은 config에서 가져옴
            conf_threshold (float, optional): 신뢰도 임계값. 기본값은 config에서 가져옴
        """
        self.model_path = model_path or config.VEHICLE_DETECTION_MODEL
        self.conf_threshold = conf_threshold or config.VEHICLE_DETECTION_CONF
        
        # YOLO 모델 로드
        self.model = YOLO(self.model_path)
        
        # GPU 사용 가능 시 GPU 사용
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Vehicle Detector using device: {self.device}")
    
    def detect(self, image):
        """
        이미지에서 차량 검출
        
        Args:
            image (numpy.ndarray): BGR 형식의 이미지
            
        Returns:
            list: 검출된 차량의 바운딩 박스 목록 [x1, y1, x2, y2]
        """
        # 이미지 크기 조정 및 정규화
        processed_img = cv2.resize(image, config.IMAGE_SIZE)
        
        # YOLO 모델로 객체 감지
        results = self.model(processed_img, conf=self.conf_threshold, device=self.device)
        
        # 'car', 'truck', 'bus' 클래스에 해당하는 바운딩 박스만 추출
        vehicle_classes = [2, 5, 7]  # COCO 데이터셋 기준 차량 클래스 ID
        
        boxes = []
        for result in results:
            for box in result.boxes:
                if int(box.cls) in vehicle_classes:
                    # 원본 이미지 크기에 맞게 바운딩 박스 조정
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 크기 비율 조정
                    h, w = image.shape[:2]
                    h_ratio, w_ratio = h / config.IMAGE_SIZE[1], w / config.IMAGE_SIZE[0]
                    
                    x1, y1 = int(x1 * w_ratio), int(y1 * h_ratio)
                    x2, y2 = int(x2 * w_ratio), int(y2 * h_ratio)
                    
                    boxes.append([x1, y1, x2, y2])
        
        return boxes
    
    def visualize(self, image, boxes):
        """
        이미지에 검출된 차량의 바운딩 박스 시각화
        
        Args:
            image (numpy.ndarray): BGR 형식의 이미지
            boxes (list): 바운딩 박스 목록 [x1, y1, x2, y2]
            
        Returns:
            numpy.ndarray: 바운딩 박스가 그려진 이미지
        """
        vis_img = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, "Vehicle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return vis_img