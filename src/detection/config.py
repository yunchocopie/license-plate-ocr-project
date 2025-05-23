# Plate Detection 관련
PLATE_DETECTION_MODEL = 'path/to/your/plate_detection_model.pt'  # 사용하는 번호판 검출 모델 경로
# !!! 중요: 이 모델이 지금 테스트하려는 번호판(예: 독일 번호판)을 포함하여 학습되었는지 확인하세요.
# !!! 아니라면, 해당 번호판 유형을 검출할 수 있는 모델로 교체해야 합니다.

PLATE_DETECTION_CONF = 0.01  # 번호판 검출 신뢰도 임계값
# !!! 만약 번호판 검출이 잘 안 되면 이 값을 0.1 또는 0.05 등으로 낮춰서 테스트해보세요.

# Vehicle Detection 관련
VEHICLE_DETECTION_MODEL = 'yolov8s.pt' # 예시: COCO로 사전 학습된 모델
VEHICLE_DETECTION_CONF = 0.3
IMAGE_SIZE = (640, 640) # YOLO 입력 이미지 크기 (VehicleDetector에서 사용)