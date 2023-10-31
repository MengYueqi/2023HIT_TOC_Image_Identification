import os

from ultralytics import YOLO

model = YOLO("./runs/classify/train2/weights/last.pt")
model.train(data='./dataset', model='yolov8n-cls.yaml', epochs=30, batch=8, imgsz=1280, save=True, workers=0,
            resume=True)
