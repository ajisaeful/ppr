import os
import cv2
import math
from ultralytics import YOLO
import io
import base64
from PIL import Image
import time

def process_image(img):
    model = YOLO("YOLO-Weights/best.pt")
    classNames = ['Boots', 'Gloves', 'Helmet', 'Mask', 'Protective Glasses', 'Vest', 'fire', 'smoke']
    
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            if class_name == 'Boots':
                color = (0, 204, 255)
            elif class_name == "Gloves":
                color = (222, 82, 175)
            elif class_name == "Helmet":
                color = (0, 149, 255)
            elif class_name == "Mask":
                color = (255, 255, 0)
            elif class_name == "Protective Glasses":
                color = (255, 0, 0)
            elif class_name == "Vest":
                color = (255, 165, 0)
            elif class_name == "fire":
                color = (255, 125, 0)
            elif class_name == "smoke":
                color = (255, 145, 0)
            else:
                color = (85, 45, 255)
            
            if conf > 0.3:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    return img