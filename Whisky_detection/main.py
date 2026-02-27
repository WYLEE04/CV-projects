"""
main.py — 위스키 라벨 스캐너 터미널 버전
Usage:
    python3 main.py --image whisky.jpg
    python3 main.py --camera
"""

import argparse
import cv2
import numpy as np
from ocr import WhiskyOCR
from matcher import WhiskyMatcher


def parse_args():
    parser = argparse.ArgumentParser(description="위스키 라벨 스캐너")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="이미지 경로")
    group.add_argument("--camera", action="store_true", help="웹캠 모드")
    parser.add_argument("--db", type=str, default="scotch_whisky.csv")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def draw_overlay(image, result):
    h, w = image.shape[:2]
    canvas = np.zeros((h + 110, w, 3), dtype=np.uint8)
    canvas[:h] = image
    canvas[h:] = (30, 30, 30)

    if result["matched"]:
        wr = result["best"]
        score = result["score"]
        cv2.putText(canvas, f"{wr['Distillery']} ({wr['Region']})",
                    (10, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        # 레이더 수치 미니 바
        from matcher import TASTING_COLS, TASTING_KO
        cols = TASTING_COLS[:6]
        for i, col in enumerate(cols):
            val = int(wr.get(col, 0))
            x = 10 + i * (w // 6)
            cv2.putText(canvas, TASTING_KO[col][:3], (x, h + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 145, 122), 1)
            cv2.rectangle(canvas, (x, h + 62), (x + (w // 6) - 8, h + 68), (50, 40, 20), -1)
            cv2.rectangle(canvas, (x, h + 62), (x + int(((w // 6) - 8) * val / 4), h + 68),
                          (201, 168, 76), -1)
        cv2.putText(canvas, f"Match: {score:.0%}",
                    (w - 120, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    else:
        cv2.putText(canvas, "매칭 실패 — 라벨을 더 선명하게 촬영해주세요",
                    (10, h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 2)
    return canvas


def run_image(args, ocr, matcher):
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"이미지 없음: {args.image}")

    full_text, ocr_results = ocr.read_label(image)
    result = matcher.match(full_text)

    print("\n" + "=" * 55)
    print(matcher.format_result(result))
    print("=" * 55)

    annotated = ocr.visualize(image, ocr_results)
    final = draw_overlay(annotated, result)
    cv2.imshow("Whisky Scanner", cv2.resize(final, (min(900, final.shape[1]), 
               int(final.shape[0] * min(900, final.shape[1]) / final.shape[1]))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(args, ocr, matcher):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠 없음")

    print("[카메라] Space: 스캔 / ESC: 종료")
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = w//4, h//6, 3*w//4, 5*h//6
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (201, 168, 76), 2)
        cv2.putText(display, "라벨을 박스 안에 위치 후 SPACE",
                    (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (201, 168, 76), 1)

        if last_result:
            display = draw_overlay(display, last_result)

        cv2.imshow("Whisky Scanner (Space/ESC)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == 32:
            roi = frame[by1:by2, bx1:bx2]
            full_text, ocr_results = ocr.read_label(roi)
            last_result = matcher.match(full_text)
            print(matcher.format_result(last_result))

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    ocr = WhiskyOCR(gpu=args.gpu)
    matcher = WhiskyMatcher(csv_path=args.db)

    if args.image:
        run_image(args, ocr, matcher)
    else:
        run_camera(args, ocr, matcher)


if __name__ == "__main__":
    main()