"""
ocr.py — 이미지 전처리 + EasyOCR 텍스트 인식
"""

import cv2
import numpy as np
from typing import List, Tuple
import easyocr


class WhiskyOCR:
    def __init__(self, languages=["en"], gpu=False, min_conf=0.3):
        print("[OCR] EasyOCR 초기화 중...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.min_conf = min_conf
        print("[OCR] 초기화 완료")

    def read_label(self, image: np.ndarray) -> Tuple[str, List[dict]]:
        processed = self._preprocess(image)
        raw_orig = self.reader.readtext(image)
        raw_proc = self.reader.readtext(processed)
        merged = self._merge_results(raw_orig, raw_proc)

        results = []
        texts = []
        for bbox, text, conf in merged:
            if conf >= self.min_conf and len(text.strip()) > 0:
                results.append({"text": text.strip(), "confidence": round(conf, 3), "bbox": bbox})
                texts.append(text.strip())

        return " ".join(texts), results

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(binary == 255) / binary.size < 0.3:
            binary = cv2.bitwise_not(binary)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    def _merge_results(self, r1, r2):
        seen = {}
        for bbox, text, conf in r1 + r2:
            key = text.strip().lower()
            if key not in seen or seen[key][2] < conf:
                seen[key] = (bbox, text.strip(), conf)
        return list(seen.values())

    def visualize(self, image: np.ndarray, results: List[dict]) -> np.ndarray:
        vis = image.copy()
        for r in results:
            pts = np.array(r["bbox"], dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            x, y = pts[0]
            cv2.putText(vis, f"{r['text']} ({r['confidence']:.2f})",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis