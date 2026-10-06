"""Does the pipeline survive transformations a new image set might apply? (REPORT.md section 5)

    python experiments/augmentation_stress.py

Development images (4 per class by default) are each transformed 13 ways and keep
their label: the caption is moved or degraded, not rewritten. Two transformations
remove caption pixels on purpose (occlude, crop_band), so a miss there means the
drink word is gone. Images are saved to data/augmented/.
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from common import CACHE, ROOT, image_path, load_split, pipeline, table, write_report
from src.metrics import binary_metrics
from src.ocr import make_variant

OUT = ROOT / "data" / "augmented"


def perspective(image: Image.Image) -> Image.Image:
    """Pull the frame into a trapezoid, so the caption is skewed rather than rotated."""
    w, h = image.size
    dx, dy = 0.18 * w, 0.08 * h
    return image.transform(image.size, Image.QUAD, (dx, dy, 0, h - 1, w - 1, h - 1, w - 1 - dx, dy),
                           resample=Image.BICUBIC)


def occlude(image: Image.Image) -> Image.Image:
    """Cover the middle of the lower band, where captions are often drawn."""
    out = image.copy()
    w, h = out.size
    ImageDraw.Draw(out).rectangle([int(w * 0.28), int(h * 0.58), int(w * 0.72), int(h * 0.80)],
                                  fill=(24, 24, 24))
    return out


def crop_band(image: Image.Image) -> Image.Image:
    """Cut out the lower band and close the gap."""
    w, h = image.size
    y0, y1 = int(h * 0.55), min(h - 1, int(h * 0.82))
    top, bottom = image.crop((0, 0, w, y0)), image.crop((0, y1, w, h))
    out = Image.new("RGB", (w, top.height + bottom.height))
    out.paste(top, (0, 0))
    out.paste(bottom, (0, top.height))
    return out


def tiny(image: Image.Image) -> Image.Image:
    small = image.resize((max(1, image.width // 3), max(1, image.height // 3)), Image.LANCZOS)
    return small.resize(image.size, Image.LANCZOS)


def jpeg(image: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=12)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


TRANSFORMS = {
    "rot60": lambda im: make_variant(im, "rot60"),
    "rot90": lambda im: make_variant(im, "rot90"),
    "upside_down": lambda im: make_variant(im, "rot180"),
    "mirror_and_30": lambda im: make_variant(im, "mirror_rot30"),
    "mirror_and_60": lambda im: make_variant(im, "mirror_rot60"),
    "perspective": perspective,
    "occlude": occlude,
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(max(1.2, min(im.size) * 0.008))),
    "crop_band": crop_band,
    "low_contrast": lambda im: ImageEnhance.Contrast(im).enhance(0.35),
    "tiny": tiny,
    "jpeg": jpeg,
    "washed": lambda im: ImageOps.equalize(ImageEnhance.Color(im).enhance(0.2)),
}


def build(per_class: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    dev = load_split("dev")
    picked = [r for label in ("alcohol", "non_alcohol")
              for r in rng.sample([d for d in dev if d["label"] == label], per_class)]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in picked:
        source = Image.open(image_path(r)).convert("RGB")
        for name, transform in TRANSFORMS.items():
            file_name = f"{Path(r['file_name']).stem}__{name}.jpg"
            transform(source).save(OUT / file_name, quality=90)
            rows.append({"file_name": file_name, "source": r["file_name"], "label": r["label"],
                         "transform": name})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_class", type=int, default=4)
    args = parser.parse_args()
    rows = build(args.per_class)
    pipe = pipeline(cache=CACHE / "augmented")
    for r in rows:
        r["prediction"] = pipe.run(str(OUT / r["file_name"])).prediction.label

    def summary(name, part):
        m = binary_metrics([r["label"] for r in part], [r["prediction"] for r in part])
        return {"transform": name, "images": len(part), "precision": m["precision"],
                "recall": m["recall"], "missed": m["fn"], "false alarms": m["fp"]}

    by_transform = [summary(t, [r for r in rows if r["transform"] == t]) for t in TRANSFORMS]
    misses = [f"- `{r['file_name']}`: true {r['label']}, predicted {r['prediction']}"
              for r in rows if r["prediction"] != r["label"]]
    write_report("augmentation_stress.md", f"Augmentation stress test ({len(rows)} images)", [
        table([summary("all", rows)]),
        "## By transformation", table(by_transform),
        "## Errors", "\n".join(misses) or "No errors.",
    ])


if __name__ == "__main__":
    main()
