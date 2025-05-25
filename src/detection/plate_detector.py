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