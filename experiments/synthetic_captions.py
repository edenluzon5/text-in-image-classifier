"""Does the pipeline work on captions it was not built around? (REPORT.md section 5)

    python experiments/synthetic_captions.py

Renders new captions onto non-alcohol development photos whose own text has no
alcohol word, then runs the final pipeline on them. Three kinds of caption:
* drinks and brands that never appear in the dataset (tanqueray, prosecco, hennessy…);
* negated alcohol (ginger ale, alcohol-free beer, mocktail…);
* non-alcohol drinks.
Each caption gets one distortion (plain, mirrored, rotated, blurred, heavy JPEG,
low contrast, leetspeak, split word), so accuracy can be read per distortion.
Images are saved to data/synthetic/.
"""

from __future__ import annotations

import argparse
import io
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from common import (CACHE, ROOT, image_path, load_split, pipeline, table, texts,
                    write_report)
from src.classify import AlcoholClassifier
from src.metrics import binary_metrics

OUT = ROOT / "data" / "synthetic"
SEED = 7
ALCOHOL_UNSEEN = ["tanqueray", "prosecco", "hennessy", "sangria", "jagermeister", "tequila",
                  "merlot", "mojito", "cognac", "pilsner", "absinthe", "martini", "riesling",
                  "captain morgan", "grey goose"]
NEGATED = ["ginger ale", "root beer", "alcohol-free beer", "non-alcoholic wine", "mocktail",
           "ginger beer"]
NON_ALCOHOL = ["kombucha", "smoothie", "orange juice", "green tea", "cold brew coffee", "soda",
               "mineral water", "hot cocoa", "latte", "protein shake"]
KINDS = [("alcohol", ALCOHOL_UNSEEN, "unseen drink or brand"),
         ("non_alcohol", NEGATED, "negated alcohol"),
         ("non_alcohol", NON_ALCOHOL, "non-alcohol drink")]
TEMPLATES = ["Try {d}", "Don't miss {d} event", "Enjoy premium {d}", "Check out {d} menu",
             "Get ready for {d} party", "Limited edition {d}", "Grab now {d} discount"]
DISTORTIONS = ["plain", "mirrored", "rotated", "blur", "jpeg", "low_contrast", "leet",
               "split_word"]
LEET = str.maketrans({"a": "4", "e": "3", "o": "0", "t": "7"})
FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Supplemental/Verdana.ttf",
         "/System/Library/Fonts/Supplemental/Georgia.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def caption(drink: str, distortion: str, rng: random.Random) -> str:
    if distortion == "leet":
        drink = drink.translate(LEET)
    if distortion == "split_word" and len(drink) > 5:
        cut = rng.randint(2, len(drink) - 3)
        drink = drink[:cut] + " " + drink[cut:]
    return rng.choice(TEMPLATES).format(d=drink)


def render(background: Image.Image, text: str, distortion: str, rng: random.Random) -> Image.Image:
    img = background.convert("RGB").copy()
    img.thumbnail((640, 640))
    fonts = [f for f in FONTS if Path(f).exists()]
    size = max(20, img.width // (22 if distortion == "rotated" else 16))
    font = ImageFont.truetype(rng.choice(fonts), size) if fonts else ImageFont.load_default(size)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    w, h = x1 - x0, y1 - y0
    x = max(5, (img.width - w) // 2)
    y = ((img.height - h) // 2 if distortion == "rotated"
         else rng.randint(img.height // 5, max(img.height // 5 + 1, img.height - h - 20)))
    if distortion == "low_contrast":
        fill = (200, 200, 200, 255)
    else:
        fill = rng.choice([(220, 30, 30, 255), (30, 90, 200, 255), (20, 20, 20, 255)])
        draw.rectangle([x - 8, y - 6, x + w + 8, y + h + 12], fill=(255, 255, 255, 235))
    draw.text((x, y), text, font=font, fill=fill)
    if distortion == "mirrored":
        layer = ImageOps.mirror(layer)
    if distortion == "rotated":
        layer = layer.rotate(rng.choice([-30, -25, 25, 30]), resample=Image.BICUBIC)
    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    if distortion == "blur":
        img = img.filter(ImageFilter.GaussianBlur(1.6))
    if distortion == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=12)
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


def build(per_cell: int, max_backgrounds: int | None) -> list[dict]:
    rng = random.Random(SEED)
    clf = AlcoholClassifier()
    candidates = load_split("dev")
    candidates = [r for r in candidates if r["label"] == "non_alcohol"][:max_backgrounds]
    backgrounds = [image_path(r) for r in candidates
                   if not clf.lexicon_evidence(texts(r, ("original", "mirror")))[0]]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, words, kind in KINDS:
        for distortion in DISTORTIONS:
            for _ in range(per_cell):
                text = caption(rng.choice(words), distortion, rng)
                name = f"syn_{len(rows):04d}.jpg"
                render(Image.open(rng.choice(backgrounds)), text, distortion, rng).save(
                    OUT / name, quality=92)
                rows.append({"file_name": name, "label": label, "kind": kind,
                             "distortion": distortion, "caption": text})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_cell", type=int, default=5,
                        help="captions per kind and distortion (5 -> 120 images)")
    parser.add_argument("--max_backgrounds", type=int, default=None,
                        help="photos to consider as backgrounds (a quick check)")
    args = parser.parse_args()
    rows = build(args.per_cell, args.max_backgrounds)
    pipe = pipeline(cache=CACHE / "synthetic")
    for r in rows:
        out = pipe.run(str(OUT / r["file_name"]))
        r["prediction"], r["reason"] = out.prediction.label, out.prediction.reason
        r["correct"] = r["prediction"] == r["label"]

    m = binary_metrics([r["label"] for r in rows], [r["prediction"] for r in rows])
    by_kind = defaultdict(list)
    by_cell = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r["correct"])
        by_cell[(r["distortion"], r["kind"])].append(r["correct"])
    kind_rows = [{"caption kind": k, "images": len(v), "accuracy": sum(v) / len(v)}
                 for k, v in by_kind.items()]
    cell_rows = [{"distortion": d, **{k: sum(by_cell[(d, k)]) / len(by_cell[(d, k)])
                                      for _, _, k in KINDS}} for d in DISTORTIONS]
    errors = [{k: r[k] for k in ("file_name", "kind", "distortion", "caption", "prediction", "reason")}
              for r in rows if not r["correct"]]
    write_report("synthetic_captions.md", f"Synthetic captions ({len(rows)} images)", [
        f"Recall {m['recall']:.3f}, precision {m['precision']:.3f}, F1 {m['f1']:.3f} "
        f"(found {m['tp']}, missed {m['fn']}, false alarms {m['fp']}).",
        "## Accuracy by caption kind", table(kind_rows),
        "## Accuracy by distortion", table(cell_rows),
        "## Errors", table(errors) if errors else "No errors.",
    ])


if __name__ == "__main__":
    main()
