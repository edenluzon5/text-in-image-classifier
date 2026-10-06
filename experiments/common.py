"""Shared helpers for the experiments: data, cached OCR, metrics, and markdown output.

Each experiment writes one markdown file to results/experiments/. PaddleOCR output
is cached in cache/, the same folder evaluate.py uses, so the images are read once.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from src.metrics import binary_metrics  # noqa: E402
from src.ocr import CachedOcr, OcrResult  # noqa: E402
from src.pipeline import AlcoholPipeline  # noqa: E402

IMAGES = ROOT / "data" / "images"
OUT_DIR = ROOT / "results" / "experiments"
CACHE = ROOT / "cache"
_paddle = CachedOcr(str(CACHE))


def limit_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N images, both classes mixed (a quick check)")
    return parser


def _read_csv(name: str) -> list[dict]:
    with open(ROOT / "data" / name, newline="") as f:
        return list(csv.DictReader(f))


def _take(rows: list[dict], limit: int | None) -> list[dict]:
    """The first ``limit`` rows, alternating the two classes."""
    if limit is None:
        return rows
    by_label = [[r for r in rows if r["label"] == lbl] for lbl in ("alcohol", "non_alcohol")]
    mixed = [r for pair in zip(*by_label) for r in pair]
    return mixed[:limit]


def load_split(*names: str, limit: int | None = None) -> list[dict]:
    return _take([r for r in _read_csv("splits.csv") if r["split"] in names], limit)


def load_gold(limit: int | None = None) -> list[dict]:
    """The 79 hand-transcribed captions (caption, drink keyword, distortion tags)."""
    return _take(_read_csv("gold_transcripts.csv"), limit)


def image_path(row: dict) -> str:
    return str(IMAGES / row["label"] / row["file_name"])


def ocr(row: dict, variant: str) -> OcrResult:
    return _paddle.read(image_path(row), variant)


def texts(row: dict, variants, min_score: float = 0.5) -> dict[str, str]:
    return {v: ocr(row, v).text(min_score) for v in variants}


def pipeline(cache: Path = CACHE, **overrides) -> AlcoholPipeline:
    """The final pipeline from config.json; ``overrides`` replace constructor arguments."""
    pipe = AlcoholPipeline.from_config(ROOT / "config.json", ocr=CachedOcr(str(cache)))
    for key, value in overrides.items():
        setattr(pipe, key, tuple(value) if key.endswith("variants") else value)
    return pipe


def metric_row(name: str, y_true, y_pred, first_col: str = "system") -> dict:
    m = binary_metrics(y_true, y_pred)
    return {first_col: name, "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "missed": m["fn"], "false alarms": m["fp"]}


def table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0])

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def write_report(name: str, title: str, sections: list[str]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text("\n\n".join([f"# {title}", *sections]) + "\n")
    print(path.read_text())
    return path


# Text comparison against the hand transcripts (no leetspeak fixing).

def light_clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate of ``hypothesis`` against ``reference``."""
    ref, hyp = light_clean(reference), light_clean(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def token_recall(reference: str, hypothesis: str) -> float:
    """Share of caption words found in the OCR text (extra scene text is not penalised)."""
    ref = light_clean(reference).split()
    hyp = set(light_clean(hypothesis).split())
    return sum(t in hyp for t in ref) / len(ref) if ref else 1.0


def keyword_survived(keyword: str, hypothesis: str) -> bool:
    """Is the caption's drink word in the OCR text, spacing ignored?"""
    k = re.sub(r"[^a-z]", "", keyword.lower())
    return bool(k) and k in re.sub(r"[^a-z]", "", hypothesis.lower())
