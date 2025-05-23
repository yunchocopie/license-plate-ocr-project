import cv2
import numpy as np
from ultralytics import YOLO
import torch
import config

"""
번호판 검출 모듈

이 모듈은 차량 이미지에서 번호판을 검출하는 클래스를 제공합니다.
"""
class PlateDetector:
    """번호판 검출을 위한 클래스"""

    def __init__(self, model_path=None, conf_threshold=None):
        """
        PlateDetector 클래스 초기화

        Args:
            model_path (str, optional): YOLO 모델 경로. 기본값은 config에서 가져옴
            conf_threshold (float, optional): 신뢰도 임계값. 기본값은 config에서 가져옴
                                            테스트를 위해 이 값을 낮춰보세요. 예: 0.1 또는 0.05
        """
        self.model_path = model_path or config.PLATE_DETECTION_MODEL
        self.conf_threshold = conf_threshold or config.PLATE_DETECTION_CONF

        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading YOLO plate detection model from {self.model_path}: {e}")
            print("Please ensure the model path is correct and the model file is accessible.")
            self.model = None
            return

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Plate Detector using device: {self.device}")
        if self.model:
            print(f"Plate Detector model loaded. Class names: {self.model.names}")
            print("IMPORTANT: Check the class names above. The expected plate class name (e.g., 'plate', 'license_plate') MUST match one of these.")
            # 클래스 이름 목록에서 'plate' 또는 'license_plate' 등이 있는지 확인하고
            # self.target_plate_class_name 같은 인스턴스 변수로 저장해두는 것도 좋습니다.
            # 예: self.target_plate_class_name = 'plate' # 실제 모델의 클래스 이름으로 변경

    def detect(self, image):
        """
        이미지에서 번호판 검출

        Args:
            image (numpy.ndarray): BGR 형식의 차량 이미지 (또는 전체 이미지)

        Returns:
            list: 검출된 번호판의 바운딩 박스 목록 [x1, y1, x2, y2]
        """
        if self.model is None:
            print("PlateDetector: Model not loaded. Cannot detect plates.")
            return []
        if image is None or image.size == 0:
            print("PlateDetector: Input image is empty.")
            return []

        # !!! 중요: 이 부분을 모델의 실제 번호판 클래스 이름으로 설정하세요.
        # 생성자에서 self.model.names를 보고 결정하거나, 인스턴스 변수로 저장한 값을 사용합니다.
        # 예시로 'plate'를 사용합니다. 실제 모델에 맞게 수정해야 합니다.
        expected_plate_class_name = 'plate' # <--- 이 부분을 메서드 상단으로 이동 및 초기화

        try:
            results = self.model(image, conf=self.conf_threshold, device=self.device, verbose=False)
        except Exception as e:
            print(f"Error during YOLO model prediction for plate detection: {e}")
            return []

        detected_boxes = []
        print(f"\n--- PlateDetector Detections on image (shape: {image.shape}, conf_thresh: {self.conf_threshold}) ---")
        for i, r_result in enumerate(results):
            if r_result.boxes:
                for box_idx, box_obj in enumerate(r_result.boxes):
                    cls_id = int(box_obj.cls[0])
                    class_name_from_model = self.model.names[cls_id]
                    confidence = box_obj.conf[0].item()
                    x1, y1, x2, y2 = map(int, box_obj.xyxy[0].tolist())

                    print(f"  Detected Object {box_idx}: Class='{class_name_from_model}' (ID:{cls_id}), Confidence={confidence:.3f}, Coords=[{x1},{y1},{x2},{y2}]")

                    if class_name_from_model.lower() == expected_plate_class_name.lower():
                        detected_boxes.append([x1, y1, x2, y2])
                        print(f"    -> Matched as '{expected_plate_class_name}' and added.")
            else:
                print("  No objects (boxes) detected in this result batch.")
        print("--------------------------------------------------------------------")

        # 이 부분에서 오류가 발생했었습니다.
        if not detected_boxes:
            print(f"No '{expected_plate_class_name}' found with current settings.")
        return detected_boxes

    def detect_best_plate(self, image):
        """
        이미지에서 신뢰도가 가장 높은 번호판 하나만 검출하여 크롭된 이미지를 반환.
        """
        if self.model is None:
            print("PlateDetector (best_plate): Model not loaded.")
            return None
        if image is None or image.size == 0:
            print("PlateDetector (best_plate): Input image is empty.")
            return None

        # !!! 중요: 이 부분도 모델의 실제 번호판 클래스 이름으로 설정하세요.
        expected_plate_class_name = 'plate' # <--- 메서드 상단으로 이동 및 초기화

        try:
            results = self.model(image, conf=self.conf_threshold, device=self.device, verbose=False)
        except Exception as e:
            print(f"Error during YOLO model prediction for best_plate: {e}")
            return None

        best_plate_object = None
        highest_confidence = -1.0

        for r_result in results:
            if r_result.boxes:
                for box_obj in r_result.boxes:
                    cls_id = int(box_obj.cls[0])
                    class_name_from_model = self.model.names[cls_id]
                    confidence = box_obj.conf[0].item()

                    if class_name_from_model.lower() == expected_plate_class_name.lower():
                        if confidence > highest_confidence:
                            highest_confidence = confidence
                            best_plate_object = box_obj

        if best_plate_object:
            x1, y1, x2, y2 = map(int, best_plate_object.xyxy[0].tolist())
            h_img, w_img = image.shape[:2]
            x1_c = max(0, x1)
            y1_c = max(0, y1)
            x2_c = min(w_img, x2)
            y2_c = min(h_img, y2)

            if x1_c < x2_c and y1_c < y2_c :
                cropped_plate = image[y1_c:y2_c, x1_c:x2_c]
                print(f"Best plate found and cropped: Class='{self.model.names[int(best_plate_object.cls[0])]}', Conf={highest_confidence:.3f}, Coords=[{x1_c},{y1_c},{x2_c},{y2_c}]")
                return cropped_plate
            else:
                print(f"Best plate found but coordinates [{x1_c},{y1_c},{x2_c},{y2_c}] are invalid for cropping. Original box: [{x1},{y1},{x2},{y2}]")
                return None
        else:
            print(f"No '{expected_plate_class_name}' found that meets the criteria for 'best_plate'.")
            return None

    # visualize 와 crop_plates 메서드는 이전과 동일하게 유지
    def visualize(self, image, boxes_coords_list):
        vis_img = image.copy()
        for box_coords in boxes_coords_list:
            x1, y1, x2, y2 = box_coords
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_img, "Plate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return vis_img

    def crop_plates(self, image, boxes_coords_list, padding=5):
        h_img, w_img = image.shape[:2]
        plate_images = []
        if not boxes_coords_list:
            return plate_images
        for box_coords in boxes_coords_list:
            x1, y1, x2, y2 = box_coords
            x1_p = max(0, x1 - padding)
            y1_p = max(0, y1 - padding)
            x2_p = min(w_img, x2 + padding)
            y2_p = min(h_img, y2 + padding)
            if x1_p < x2_p and y1_p < y2_p:
                plate_img = image[y1_p:y2_p, x1_p:x2_p]
                if plate_img.size > 0:
                    plate_images.append(plate_img)
        return plate_images