from ultralytics import YOLO

model = YOLO('best.pt')
model.predict(source='testset/', save=True, project="./")