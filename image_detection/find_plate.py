import os
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(MODEL_PATH)

# 이미지에서 번호판을 찾는 함수
def detect_plate(img):
    results = model.predict(img, conf=0.5) # 신뢰도 0.5 이상의 번호판을 찾음

    for r in results:
        boxes = r.boxes # 감지된 객체들의 바운딩 박스들
        best_plate = None # 가장 높은 확신도의 박스를 저장하기 위한 변수
        best_conf = -1 # 가장 높은 신뢰도 저장

        for box in boxes: # 가장 높은 신뢰도의 번호판을 찾기 위한 반복문
            cls_id = int(box.cls[0]) # 0번 클래스(번호판) 아이디
            class_name = model.names[cls_id] # 클래스 이름(plate) 가져오기

            if class_name != 'plate': # 번호판이 아닌 클래스는 무시
                continue

            conf = box.conf[0].item()  # 현재 박스의 신뢰도
            if conf > best_conf: # 가장 높은 신뢰도의 박스를 찾기
                best_conf = conf
                best_plate = box

        if best_plate: # 가장 신뢰도가 높은 번호판이 있다면 크롭
            x1, y1, x2, y2 = map(int, best_plate.xyxy[0].tolist())
            cropped = img[y1:y2, x1:x2]

            return cropped
    return None

