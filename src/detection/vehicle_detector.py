import cv2
import numpy as np
from ultralytics import YOLO
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
            image (numpy.ndarray): BGR 형식의 단일 이미지 (배치 입력 미지원)

        Returns:
            list: 검출된 차량의 바운딩 박스 목록 [x1, y1, x2, y2] (원본 이미지 좌표)
        """
        # Ultralytics 내부 letterbox 활용으로 종횡비 유지
        # imgsz 파라미터를 전달하면 자동으로 letterbox 패딩 적용
        results = self.model(
            image,
            imgsz=config.IMAGE_SIZE[0],  # 단일 값으로 전달 (640)
            conf=self.conf_threshold,
            device=self.device,
            verbose=False  # 진행 로그 숨김
        )

        # 'car', 'truck', 'bus' 클래스에 해당하는 바운딩 박스만 추출
        vehicle_classes = [2, 5, 7]  # COCO 데이터셋 기준 차량 클래스 ID

        boxes = []
        # 단일 이미지 처리: results[0]만 사용
        result = results[0]
        for box in result.boxes:
            if int(box.cls) in vehicle_classes:
                # Ultralytics는 자동으로 원본 이미지 좌표로 반환
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([int(x1), int(y1), int(x2), int(y2)])

        return boxes
