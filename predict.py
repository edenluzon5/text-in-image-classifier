"""Classify every image in a directory as alcohol / non_alcohol.

    python predict.py --input_dir path/to/images --output results.csv

Writes ``file_name,prediction`` (``alcohol`` / ``non_alcohol``), one row per
image, in the format of ``data/example_results.csv``. Subfolders are included.
The file name is only an identifier; the prediction comes from the pixels.

Optional:
    --details details.csv   also write confidence, reason, matched words and
                            the OCR text of every image
    --cache_dir DIR         keep the OCR output so a re-run is instant
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from src.classify import NON_ALCOHOL  # noqa: E402
from src.ocr import CachedOcr  # noqa: E402
from src.pipeline import AlcoholPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def classify_image(pipe: AlcoholPipeline, path: Path) -> dict:
    """One output row. An unreadable file is kept and labelled non_alcohol."""
    try:
        out = pipe.run(str(path))
    except Exception as exc:
        logging.warning("failed on %s: %s", path, exc)
        return {"file_name": path.name, "prediction": NON_ALCOHOL, "confidence": "low",
                "reason": f"error: {exc}", "evidence": "", "ocr_text": ""}
    pred = out.prediction
    return {
        "file_name": path.name, "prediction": pred.label, "confidence": pred.confidence,
        "reason": pred.reason,
        "evidence": "; ".join(sorted({f"{m.term}<-{m.matched_text}" for m in pred.evidence})),
        "ocr_text": " | ".join(f"{k}: {v}" for k, v in out.texts.items() if v),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output", default="results.csv", type=Path)
    ap.add_argument("--details", type=Path, default=None)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    images = find_images(args.input_dir)
    if not images:
        logging.error("no images found under %s", args.input_dir)
        return 1

    pipe = AlcoholPipeline.from_config(ROOT / "config.json", ocr=CachedOcr(args.cache_dir))
    rows, start = [], time.time()
    for i, path in enumerate(images, 1):
        rows.append(classify_image(pipe, path))
        if i % 25 == 0 or i == len(images):
            logging.info("%d/%d images, %.1f s/image", i, len(images), (time.time() - start) / i)

    write_csv(args.output, rows, ["file_name", "prediction"])
    if args.details:
        write_csv(args.details, rows, list(rows[0]))
    n_alcohol = sum(r["prediction"] != NON_ALCOHOL for r in rows)
    logging.info("wrote %s: %d images, %d alcohol", args.output, len(rows), n_alcohol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
