"""Which OCR engine reads these captions best? (REPORT.md section 3, "OCR engine")

    python experiments/ocr_comparison.py

PaddleOCR, EasyOCR and Tesseract read the 79 hand-transcribed images, each on the
original image, + mirrored, and + vertically flipped. Measured per engine:

* drink word found  - the caption's drink word is in the OCR text (what the classifier needs)
* caption words     - share of caption words found (extra scene text is not penalised)
* CER               - character error rate of the most confident variant, median
* no text           - images where nothing was read
* seconds / image

Needs the extra packages in experiments/requirements.txt and `brew install tesseract`.
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np

from common import (CACHE, image_path, keyword_survived, limit_arg, load_gold, table,
                    token_recall, cer, write_report)
from src.ocr import CachedOcr, Detection, mean_confidence

STRATEGIES = {
    "original": ("original",),
    "+ mirror": ("original", "mirror"),
    "+ mirror + vertical flip": ("original", "mirror", "mirror_rot180"),
}


class EasyEngine:
    def __init__(self) -> None:
        import easyocr

        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, image) -> list[Detection]:
        return [Detection(text=t, score=float(s), box=np.asarray(b, dtype=float).tolist())
                for b, t, s in self._reader.readtext(np.asarray(image.convert("RGB")))
                if t.strip()]


class TesseractEngine:
    def __init__(self) -> None:
        import pytesseract

        self._pt = pytesseract

    def read(self, image) -> list[Detection]:
        data = self._pt.image_to_data(image.convert("RGB"), config="--psm 11",
                                      output_type=self._pt.Output.DICT)
        lines: dict[tuple, list[int]] = {}
        for i, word in enumerate(data["text"]):
            if word.strip() and float(data["conf"][i]) >= 0:
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines.setdefault(key, []).append(i)
        dets = []
        for idx in lines.values():
            x0 = min(data["left"][i] for i in idx)
            y0 = min(data["top"][i] for i in idx)
            x1 = max(data["left"][i] + data["width"][i] for i in idx)
            y1 = max(data["top"][i] + data["height"][i] for i in idx)
            dets.append(Detection(
                text=" ".join(data["text"][i] for i in idx),
                score=float(np.mean([float(data["conf"][i]) for i in idx])) / 100.0,
                box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        return dets


class OtherOcr(CachedOcr):
    """CachedOcr with another engine, cached in its own folder."""

    def __init__(self, name: str, factory) -> None:
        super().__init__(str(CACHE / "experiments" / name))
        self._factory = factory

    @property
    def engine(self):
        if self._engine is None:
            self._engine = self._factory()
        return self._engine


def readers() -> dict[str, CachedOcr]:
    return {"paddle": CachedOcr(str(CACHE)),
            "easyocr": OtherOcr("easyocr", EasyEngine),
            "tesseract": OtherOcr("tesseract", TesseractEngine)}


def main() -> None:
    args = limit_arg(argparse.ArgumentParser(description=__doc__)).parse_args()
    gold = load_gold(args.limit)
    rows = []
    for engine, reader in readers().items():
        for strategy, variants in STRATEGIES.items():
            found, words, cers, empty, secs = [], [], [], 0, []
            for g in gold:
                results = [reader.read(image_path(g), v) for v in variants]
                union = " ".join(r.text() for r in results)
                best = max(results, key=lambda r: mean_confidence(r.detections))
                if g["keyword"]:
                    found.append(keyword_survived(g["keyword"], union))
                words.append(token_recall(g["caption"], union))
                if g["caption"]:
                    cers.append(cer(g["caption"], best.text()))
                empty += not union.strip()
                secs.append(sum(r.seconds for r in results))
            rows.append({"engine": engine, "variants read": strategy,
                         "drink word found": sum(found) / max(1, len(found)),
                         "caption words": statistics.mean(words),
                         "CER (median)": statistics.median(cers) if cers else float("nan"),
                         "no text": empty / len(gold),
                         "seconds / image": statistics.mean(secs)})
    write_report("ocr_comparison.md", f"OCR engine comparison ({len(gold)} hand-transcribed images)", [
        "\"Drink word found\" ignores spacing and does no leetspeak fixing.",
        table(rows),
    ])


if __name__ == "__main__":
    main()
