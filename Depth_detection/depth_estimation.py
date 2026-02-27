import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from transformers import pipeline
from PIL import Image
import io

# FastAPI 앱 초기화
app = FastAPI(
    title="3D Depth Estimation API",
    description="2D 이미지를 입력받아 3D 깊이 맵(Depth Map)을 반환하는 비전 서빙 API"
)

# 1. Hugging Face 비전 모델 로드
print("딥러닝 모델을 불러오는 중입니다... (최초 실행 시 1~2분 소요)")
depth_estimator = pipeline(task="depth-estimation") 
print("모델 로드 완료! 서버가 준비되었습니다.")

@app.post("/predict-depth/")
async def predict_depth(file: UploadFile = File(...)):
    # 2. 클라이언트가 보낸 이미지 읽기
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 3. 비전 모델 추론 (Inference)
    result = depth_estimator(image)
    depth_image = result["depth"] # PIL 이미지 형태의 흑백 뎁스 맵
    
    # 4. OpenCV를 이용한 이미지 후처리 (시각화 향상)
    depth_array = np.array(depth_image)
    
    # 정규화 (0~255)
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    depth_normalized = (depth_array - depth_min) / (depth_max - depth_min) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # 컬러맵 적용 (거리가 가까울수록 붉고 밝게 표현)
    colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    
    # 5. 결과를 JPG 파일 형태로 인코딩하여 반환
    _, encoded_img = cv2.imencode('.jpg', colored_depth)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.get("/")
def read_root():
    return {"message": "3D Depth Estimation API Server is Running!"}