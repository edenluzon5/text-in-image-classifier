"""Read the text in an image with PaddleOCR, on several versions of the image.

The captions in this dataset are often mirrored, tilted, upside down, or small.
No OCR engine reads mirrored letters, and PaddleOCR's detector often finds no
text box on a steeply tilted banner. So the image is also read after it is
transformed. A variant name is a chain of operations joined by ``_``:

* ``original``      - the image as-is
* ``mirror``        - flipped left-right (fixes mirrored text)
* ``rotN``          - rotated counter-clockwise by N degrees (``rot35``, ``rot-35``)
* ``upscaleN``      - enlarged N times (separates tightly packed lines)
* ``autocontrast``  - stretched contrast (washed-out captions)
* ``sharpen``       - unsharp mask (blurred captions)

``mirror_rot35`` mirrors and then rotates.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Named explicitly so a different paddleocr default cannot change the results.
PADDLE_DET_MODEL = "PP-OCRv6_medium_det"
PADDLE_REC_MODEL = "PP-OCRv6_medium_rec"
PADDLE_TEXTLINE_MODEL = "PP-LCNet_x1_0_textline_ori"


@dataclass
class Detection:
    text: str
    score: float
    box: list[list[float]]  # polygon points, (x, y) in pixels


@dataclass
class OcrResult:
    variant: str
    width: int
    height: int
    detections: list[Detection]
    seconds: float

    def text(self, min_score: float = 0.0) -> str:
        return " ".join(d.text for d in self.detections if d.score >= min_score)

    @staticmethod
    def from_dict(d: dict) -> "OcrResult":
        return OcrResult(
            variant=d["variant"], width=d["width"], height=d["height"],
            detections=[Detection(**x) for x in d["detections"]], seconds=d["seconds"],
        )


def polygon_long_side(box: list[list[float]]) -> float:
    """Length of the longest edge of a detection polygon, in pixels."""
    if len(box) < 2:
        return 0.0
    best = 0.0
    for i, (x1, y1) in enumerate(box):
        x2, y2 = box[(i + 1) % len(box)]
        best = max(best, math.hypot(float(x2) - float(x1), float(y2) - float(y1)))
    return best


def frame_span(result: OcrResult) -> float:
    """Longer side of the frame the boxes were drawn on.

    Rotated variants store the original image size, while their boxes live in
    the expanded canvas, so the span follows the boxes when they run past the
    stored width and height.
    """
    max_x = max_y = 0.0
    for det in result.detections:
        for x, y in det.box:
            max_x = max(max_x, float(x))
            max_y = max(max_y, float(y))
    return max(float(result.width), float(result.height), max_x, max_y, 1.0)


def mean_confidence(detections: list[Detection]) -> float:
    """Character-weighted mean recognition confidence of these lines."""
    kept = [d for d in detections if d.text.strip()]
    total = sum(len(d.text) for d in kept)
    if total == 0:
        return 0.0
    return sum(d.score * len(d.text) for d in kept) / total


def _sharpen(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))


_OPS = {
    "mirror": ImageOps.mirror,
    "autocontrast": ImageOps.autocontrast,
    "sharpen": _sharpen,
}


def make_variant(image: Image.Image, variant: str) -> Image.Image:
    """Apply a variant such as ``original``, ``mirror``, ``rot35``, ``mirror_rot-35``."""
    if variant == "original":
        return image
    out = image
    for op in variant.split("_"):
        if op in _OPS:
            out = _OPS[op](out)
        elif op.startswith("rot"):
            out = out.rotate(float(op[3:]), expand=True, fillcolor=(255, 255, 255),
                             resample=Image.BICUBIC)
        elif op.startswith("upscale"):
            k = float(op[len("upscale"):])
            out = out.resize((round(out.width * k), round(out.height * k)), Image.LANCZOS)
        else:
            raise ValueError(f"unknown variant {variant}")
    return out


class PaddleEngine:
    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            text_detection_model_name=PADDLE_DET_MODEL,
            text_recognition_model_name=PADDLE_REC_MODEL,
            textline_orientation_model_name=PADDLE_TEXTLINE_MODEL,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

    def read(self, image: Image.Image) -> list[Detection]:
        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        res = self._ocr.predict(bgr)[0]
        return [
            Detection(text=t, score=float(s), box=np.asarray(p).tolist())
            for t, s, p in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"])
            if t.strip()
        ]


class CachedOcr:
    """Reads image variants with PaddleOCR, optionally caching results as JSON."""

    def __init__(self, cache_dir: str | None = None) -> None:
        self._engine = None
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @property
    def engine(self) -> PaddleEngine:
        if self._engine is None:
            self._engine = PaddleEngine()
        return self._engine

    def _cache_path(self, image_path: str, variant: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / variant / (Path(image_path).name + ".json")

    def read(self, image_path: str, variant: str = "original") -> OcrResult:
        cache = self._cache_path(image_path, variant)
        if cache is not None and cache.exists():
            return OcrResult.from_dict(json.loads(cache.read_text()))
        image = Image.open(image_path).convert("RGB")
        start = time.perf_counter()
        dets = self.engine.read(make_variant(image, variant))
        result = OcrResult(variant, image.width, image.height, dets,
                           time.perf_counter() - start)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(asdict(result)))
        return result
