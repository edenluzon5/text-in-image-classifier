"""Score the model on a labelled split and write the results to results/<split>/.

    python evaluate.py --split test          # or --split val

Needs the labelled images in data/images/alcohol and data/images/non_alcohol
(``python download_data.py``). The split of each image is in data/splits.csv.

Writes:
    results.csv           predictions in the required file_name,prediction format
    predictions.csv       the same rows plus true label, confidence, reason, OCR text
    metrics.md            precision, recall, F1, accuracy with 95% bootstrap intervals,
                          accuracy by confidence, and every error with its OCR text
    confusion_matrix.png
    errors.png            the misclassified images with what OCR read
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from predict import ROOT, classify_image, write_csv  # noqa: E402
from src.metrics import binary_metrics, bootstrap_ci  # noqa: E402
from src.ocr import CachedOcr  # noqa: E402
from src.pipeline import AlcoholPipeline  # noqa: E402


def load_split(name: str) -> list[dict]:
    with open(ROOT / "data" / "splits.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == name]


def metrics_report(split: str, rows: list[dict]) -> str:
    y_true = [r["label"] for r in rows]
    y_pred = [r["prediction"] for r in rows]
    m, ci = binary_metrics(y_true, y_pred), bootstrap_ci(y_true, y_pred)
    lines = [
        f"# Results on the {split} split (n = {len(rows)})",
        "",
        "| metric | value | 95% bootstrap interval |",
        "|---|---:|---:|",
    ]
    for k in ("precision", "recall", "f1", "accuracy"):
        lines.append(f"| {k} | {m[k]:.3f} | {ci[k][0]:.3f} – {ci[k][1]:.3f} |")
    lines += [
        "",
        f"Alcohol images found: {m['tp']}. Missed: {m['fn']}. "
        f"False alarms: {m['fp']}. Non-alcohol correct: {m['tn']}.",
        "",
        "## Accuracy by confidence",
        "",
        "| confidence | images | correct | accuracy |",
        "|---|---:|---:|---:|",
    ]
    for conf in ("high", "medium", "low"):
        sub = [r for r in rows if r["confidence"] == conf]
        if sub:
            ok = sum(r["label"] == r["prediction"] for r in sub)
            lines.append(f"| {conf} | {len(sub)} | {ok} | {ok / len(sub):.3f} |")
    errors = [r for r in rows if r["label"] != r["prediction"]]
    lines += ["", f"## Errors ({len(errors)})", ""]
    if errors:
        lines += ["| file | true label | prediction | reason | what OCR read |", "|---|---|---|---|---|"]
        for r in errors:
            ocr = (r["ocr_text"] or "(nothing)").replace("|", "/")[:160]
            lines.append(f"| {r['file_name']} | {r['label']} | {r['prediction']} | "
                         f"{r['reason']} | {ocr} |")
    return "\n".join(lines) + "\n"


def plot_confusion(rows: list[dict], path: Path, title: str) -> None:
    m = binary_metrics([r["label"] for r in rows], [r["prediction"] for r in rows])
    mat = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=15,
                color="white" if v > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["alcohol", "non_alcohol"])
    ax.set_yticks([0, 1], ["alcohol", "non_alcohol"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(rows: list[dict], images: Path, path: Path, limit: int = 9) -> None:
    errors = [r for r in rows if r["label"] != r["prediction"]][:limit]
    if not errors:
        return
    cols = min(3, len(errors))
    nrows = (len(errors) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 4.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, errors):
        ax.imshow(Image.open(images / r["label"] / r["file_name"]).convert("RGB"))
        read = r["ocr_text"].encode("ascii", "ignore").decode() or "(nothing)"
        ax.set_title(
            f"{r['file_name']}: true {r['label']}, predicted {r['prediction']}\n"
            + textwrap.shorten(f"OCR read: {read}", 110, placeholder="…"),
            fontsize=8, wrap=True,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--images", type=Path, default=ROOT / "data" / "images")
    ap.add_argument("--cache_dir", default=str(ROOT / "cache"))
    args = ap.parse_args()

    out_dir = ROOT / "results" / args.split
    pipe = AlcoholPipeline.from_config(ROOT / "config.json", ocr=CachedOcr(args.cache_dir))
    rows = []
    for i, item in enumerate(load_split(args.split), 1):
        row = classify_image(pipe, args.images / item["label"] / item["file_name"])
        rows.append({**row, "label": item["label"]})
        if i % 25 == 0:
            print(f"{i} images done")

    write_csv(out_dir / "results.csv", rows, ["file_name", "prediction"])
    write_csv(out_dir / "predictions.csv", rows,
              ["file_name", "label", "prediction", "confidence", "reason", "evidence", "ocr_text"])
    (out_dir / "metrics.md").write_text(metrics_report(args.split, rows))
    plot_confusion(rows, out_dir / "confusion_matrix.png", f"{args.split} split (n={len(rows)})")
    plot_errors(rows, args.images, out_dir / "errors.png")
    print((out_dir / "metrics.md").read_text())


if __name__ == "__main__":
    main()
