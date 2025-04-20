from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="dataset/data.yaml",  # yaml 경로
    epochs=50,                 # 학습 횟수
    imgsz=640,                 # 입력 이미지 크기
    batch=4,                   # 배치 사이즈 (GPU 메모리에 따라 조정)
    name="plate-detect",       # 저장 폴더 이름 (runs/detect/plate-detect)
    workers=2                  # 데이터 로딩 쓰레드 수
)
