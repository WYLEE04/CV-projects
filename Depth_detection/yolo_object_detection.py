import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="이미지 속 객체(사람, 자동차 등)를 찾아 네모 박스를 쳐주는 시각 인지 서버"
)

# 가장 가볍고 빠른 YOLOv8 Nano 모델 로드 
print("YOLOv8 모델을 불러오는 중입니다...")
model = YOLO('yolov8n.pt') 
print("모델 로드 완료! 객체 탐지 서버가 준비되었습니다.")

@app.post("/detect-objects/")
async def detect_objects(file: UploadFile = File(...)):
    # 1. 클라이언트가 보낸 이미지 읽기
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. PIL 이미지를 OpenCV 배열(BGR)로 변환
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 3. YOLO 모델 추론 (어떤 사물인지, 어디 있는지 박스 찾기)
    results = model(img_bgr)
    
    # 4. 찾은 객체들에 네모 박스와 이름(Label) 그려넣기
    res_plotted = results[0].plot() 
    
    # 5. 결과를 다시 JPG로 인코딩하여 반환
    _, encoded_img = cv2.imencode('.jpg', res_plotted)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")