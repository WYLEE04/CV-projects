"""
app.py — 위스키 라벨 스캐너 Flask 웹 UI
"""

import base64
import json
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

from ocr import WhiskyOCR
from matcher import WhiskyMatcher

app = Flask(__name__)

print("[App] 초기화 중...")
ocr = WhiskyOCR(gpu=False)
matcher = WhiskyMatcher(csv_path="scotch_whisky.csv")
print("[App] 준비 완료! http://localhost:5000")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "이미지 없음"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "이미지를 읽을 수 없습니다"}), 400

    # OCR
    full_text, ocr_results = ocr.read_label(image)

    # 매칭
    result = matcher.match(full_text)

    # OCR 시각화 이미지
    annotated = ocr.visualize(image, ocr_results)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(buf).decode("utf-8")

    response = {
        "matched": result["matched"],
        "ocr_text": result["ocr_text"],
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
    }

    if result["matched"]:
        w = result["best"]
        radar = matcher.get_radar_data(w)
        response["whisky"] = {
            "name": w["Distillery"],
            "region": w["Region"],
            "score": result["score"],
            "tasting": {
                "labels": radar["labels"],
                "values": radar["values"],
                "max": radar["max_value"],
            }
        }
        response["candidates"] = [
            {"name": c["whisky"]["Distillery"], "score": c["score"]}
            for c in result["candidates"]
        ]

    return jsonify(response)


@app.route("/db")
def get_db():
    return jsonify(matcher.db)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)