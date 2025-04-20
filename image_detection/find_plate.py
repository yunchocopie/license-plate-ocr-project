import os
from ultralytics import YOLO

# 1. YOLO 모델 로드
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(MODEL_PATH)

def detect_plate(img):
    results = model.predict(img, conf=0.5)

    for r in results:
        boxes = r.boxes
        best_plate = None
        best_conf = -1

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name != 'plate':
                continue

            conf = box.conf[0].item()
            if conf > best_conf:
                best_conf = conf
                best_plate = box

        if best_plate:
            x1, y1, x2, y2 = map(int, best_plate.xyxy[0].tolist())
            cropped = img[y1:y2, x1:x2]

            return cropped
    return None

