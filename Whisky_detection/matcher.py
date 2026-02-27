"""
matcher.py — OCR 텍스트 → Kaggle CSV 데이터 퍼지 매칭
=======================================================

Scotch Whisky Dataset (Kaggle) 컬럼:
    Distillery, Body, Sweetness, Smoky, Medicinal, Tobacco,
    Honey, Spicy, Winey, Nutty, Malty, Fruity, Floral
    (각 항목 0~4점)

매칭 전략:
    1. 키워드 직접 포함 (가중치 0.5)
    2. difflib 편집거리 유사도 (가중치 0.4)
    3. 토큰 Jaccard 유사도 (가중치 0.1)
"""

import csv
import re
import difflib
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# 레이더 차트에 사용할 테이스팅 항목
TASTING_COLS = ["Body", "Sweetness", "Smoky", "Medicinal", "Honey",
                "Spicy", "Winey", "Nutty", "Malty", "Fruity", "Floral"]

# 각 항목 한국어 설명
TASTING_KO = {
    "Body":      "바디감",
    "Sweetness": "달콤함",
    "Smoky":     "스모키",
    "Medicinal": "의약품향",
    "Honey":     "꿀/바닐라",
    "Spicy":     "스파이시",
    "Winey":     "와이니",
    "Nutty":     "너티",
    "Malty":     "몰티",
    "Fruity":    "프루티",
    "Floral":    "플로럴",
}

# 지역 정보 (증류소 → 지역 매핑)
REGION_MAP = {
    "Ardbeg": "Islay", "Bowmore": "Islay", "Bruichladdich": "Islay",
    "Bunnahabhain": "Islay", "Caol Ila": "Islay", "Lagavulin": "Islay",
    "Laphroaig": "Islay",
    "Glenfiddich": "Speyside", "Glenlivet": "Speyside", "Macallan": "Speyside",
    "Glenfarclas": "Speyside", "Balvenie": "Speyside", "Aberlour": "Speyside",
    "Cragganmore": "Speyside", "Glenrothes": "Speyside",
    "Glengoyne": "Highland", "Highland Park": "Islands", "Oban": "Highland",
    "Dalmore": "Highland", "Glenmorangie": "Highland", "Clynelish": "Highland",
    "Talisker": "Islands", "Jura": "Islands",
    "Auchentoshan": "Lowland", "Glenkinchie": "Lowland", "Bladnoch": "Lowland",
    "Springbank": "Campbeltown", "GlenScotia": "Campbeltown",
}


class WhiskyMatcher:
    def __init__(self, csv_path: str = "scotch_whisky.csv", top_k: int = 3, min_score: float = 0.2):
        self.db = self._load_csv(csv_path)
        self.top_k = top_k
        self.min_score = min_score
        print(f"[Matcher] Kaggle CSV 로드 완료: {len(self.db)}개 증류소")

    def match(self, ocr_text: str) -> Dict:
        cleaned = self._clean(ocr_text)
        scores = [(i, self._score(cleaned, w)) for i, w in enumerate(self.db)]
        scores.sort(key=lambda x: x[1], reverse=True)

        candidates = [
            {"whisky": self.db[i], "score": round(s, 3)}
            for i, s in scores[:self.top_k] if s >= self.min_score
        ]

        if candidates:
            return {
                "matched": True,
                "best": candidates[0]["whisky"],
                "score": candidates[0]["score"],
                "candidates": candidates,
                "ocr_text": ocr_text,
                "cleaned_text": cleaned,
            }
        return {"matched": False, "best": None, "score": 0.0,
                "candidates": [], "ocr_text": ocr_text, "cleaned_text": cleaned}

    def get_radar_data(self, whisky: dict) -> dict:
        """레이더 차트용 데이터 반환"""
        return {
            "labels": [TASTING_KO[c] for c in TASTING_COLS],
            "values": [int(whisky.get(c, 0)) for c in TASTING_COLS],
            "max_value": 4,
        }

    # ── 내부 메서드 ───────────────────────────────────────────────

    def _score(self, text: str, whisky: dict) -> float:
        name = whisky["Distillery"].lower()
        # 공백/대소문자 정규화된 이름
        name_clean = re.sub(r"[^a-z0-9]", "", name)
        text_clean = re.sub(r"[^a-z0-9]", "", text)

        # 1. 직접 포함
        kw_score = 0.0
        if name_clean in text_clean or name in text:
            kw_score = 1.0
        elif any(part in text for part in name.split() if len(part) > 3):
            kw_score = 0.6

        # 2. 편집거리
        fuzzy = difflib.SequenceMatcher(None, text_clean, name_clean).ratio()

        # 3. Jaccard
        t_tok = set(text.split())
        n_tok = set(name.lower().split())
        union = t_tok | n_tok
        jaccard = len(t_tok & n_tok) / len(union) if union else 0.0

        return 0.5 * kw_score + 0.4 * fuzzy + 0.1 * jaccard

    def _clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _load_csv(self, path: str) -> List[dict]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV 파일 없음: {path}\nKaggle에서 'scotch_whisky.csv' 다운로드 후 같은 폴더에 넣어주세요.")
        rows = []
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 숫자형 변환
                for col in TASTING_COLS:
                    try:
                        row[col] = int(row[col])
                    except (ValueError, KeyError):
                        row[col] = 0
                # 지역 정보 추가
                row["Region"] = REGION_MAP.get(row["Distillery"], "Scotland")
                rows.append(row)
        return rows

    def format_result(self, result: dict) -> str:
        if not result["matched"]:
            return f"❌ 매칭 실패\nOCR: {result['ocr_text']}"
        w = result["best"]
        radar = self.get_radar_data(w)
        lines = [
            f"✅ 매칭: {w['Distillery']} ({result['score']:.0%})",
            f"   지역: {w['Region']}",
            "",
            "📊 테이스팅 프로파일:",
        ]
        for label, val in zip(radar["labels"], radar["values"]):
            bar = "█" * val + "░" * (4 - val)
            lines.append(f"   {label:8s} [{bar}] {val}/4")
        return "\n".join(lines)