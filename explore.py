"""Data exploration figures for REPORT.md section 2, written to results/exploration/.

    python explore.py

Needs the labelled images in data/images/ (``python download_data.py``) and the
79 hand-transcribed captions in data/gold_transcripts.csv. No OCR is run.

Writes:
    dataset.png       images per class and split, and image sizes (from data/splits.csv)
    samples.png       example images, one per kind of text distortion
    distortions.png   how often each distortion appears in the transcribed captions
    trap_words.png    alcohol-sounding filler words appear in both classes
"""

from __future__ import annotations

import csv
import re
import textwrap
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from predict import ROOT  # noqa: E402
from src.lexicon import TRAPS  # noqa: E402

IMAGES = ROOT / "data" / "images"
OUT = ROOT / "results" / "exploration"
CLASSES = ("alcohol", "non_alcohol")
COLORS = {"alcohol": "#c0504d", "non_alcohol": "#4f81bd"}
SAMPLE_TAGS = ("plain", "leet", "split_word", "rotated", "mirrored", "low_contrast",
               "multiline", "upside_down")


def read_csv(name: str) -> list[dict]:
    with open(ROOT / "data" / name, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)


def plot_dataset(splits: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    names = ("dev", "val", "test")
    counts = Counter((r["split"], r["label"]) for r in splits)
    for k, label in enumerate(CLASSES):
        ax1.bar([i + 0.4 * k for i in range(3)], [counts[(s, label)] for s in names],
                width=0.4, color=COLORS[label], label=label)
    ax1.set_xticks([i + 0.2 for i in range(3)], ["development", "validation", "test"])
    ax1.set_ylabel("images")
    ax1.set_title("Images per split (balanced classes)")
    ax1.legend()

    for label in CLASSES:
        rows = [r for r in splits if r["label"] == label]
        ax2.scatter([int(r["width"]) for r in rows], [int(r["height"]) for r in rows],
                    s=6, alpha=0.4, color=COLORS[label], label=label)
    ax2.set_xlabel("width (px)")
    ax2.set_ylabel("height (px)")
    ax2.set_title("Image sizes")
    ax2.legend()
    save(fig, "dataset.png")


def plot_samples(gold: list[dict]) -> None:
    picks = []
    for i, tag in enumerate(SAMPLE_TAGS):
        rows = [r for r in gold if tag in r["distortions"].split(";") and r not in picks]
        preferred = [r for r in rows if r["label"] == CLASSES[i % 2]]
        if rows:
            picks.append((preferred or rows)[0])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, picks):
        ax.imshow(Image.open(IMAGES / r["label"] / r["file_name"]).convert("RGB"))
        ax.set_title(
            f"{r['label']}  [{r['distortions'].replace(';', ', ')}]\n"
            + textwrap.shorten(f'"{r["caption"]}"', 60, placeholder="…"),
            fontsize=10, color=COLORS[r["label"]],
        )
    save(fig, "samples.png")


def plot_distortions(gold: list[dict]) -> None:
    tags = Counter(t for r in gold for t in r["distortions"].split(";") if t)
    names = [t for t, _ in tags.most_common()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names[::-1], [tags[t] for t in names[::-1]], color="#7f7f7f")
    ax.set_xlabel(f"images (of {len(gold)} hand-transcribed)")
    ax.set_title("Text distortions in the captions (an image can have several)")
    save(fig, "distortions.png")


def plot_trap_words(gold: list[dict], top: int = 12) -> None:
    counts = Counter()
    for r in gold:
        for word in set(re.findall(r"[a-z]+", r["caption"].lower())) & TRAPS:
            counts[(word, r["label"])] += 1
    words = sorted({w for w, _ in counts}, key=lambda w: -(counts[(w, "alcohol")] + counts[(w, "non_alcohol")]))[:top]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    y = range(len(words))
    for k, label in enumerate(CLASSES):
        ax.barh([i + 0.4 * k for i in y], [counts[(w, label)] for w in words],
                height=0.4, color=COLORS[label], label=f"{label} captions")
    ax.set_yticks([i + 0.2 for i in y], words)
    ax.invert_yaxis()
    ax.set_xlabel("captions containing the word")
    ax.set_title("Alcohol-sounding filler words appear in both classes")
    ax.legend()
    save(fig, "trap_words.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gold = read_csv("gold_transcripts.csv")
    plot_dataset(read_csv("splits.csv"))
    plot_samples(gold)
    plot_distortions(gold)
    plot_trap_words(gold)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
