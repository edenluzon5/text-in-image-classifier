"""What does each part of the pipeline add, and how do trained models compare?
(REPORT.md section 4, "What each part contributes", and section 5)

    python experiments/components.py

On the validation split (220 images):
* parts of the final pipeline are added one at a time;
* character and word n-gram classifiers, trained on development OCR text, are
  scored on their own and as a fallback for images the rules leave undecided.
"""

from __future__ import annotations

import argparse
import math

from common import (image_path, limit_arg, load_split, metric_row, pipeline, table, texts,
                    write_report)
from ngram import HybridClassifier, build_char_model, build_word_model
from src.classify import AlcoholClassifier, LexiconMatcher, MatcherConfig
from src.normalize import NormalizeConfig
from src.pipeline import MAIN_VARIANTS

NO_RULES = MatcherConfig(normalize=NormalizeConfig(leet=False, confusables=False),
                         use_negations=False, use_skeleton=False, use_glue_split=False,
                         use_fuzzy=False)
NO_FUZZY = MatcherConfig(use_fuzzy=False)
NO_CAPTION_RULE = math.inf


def run(rows, **overrides) -> list[str]:
    pipe = pipeline(**overrides)
    return [pipe.run(r["path"]).prediction.label for r in rows]


def main() -> None:
    args = limit_arg(argparse.ArgumentParser(description=__doc__)).parse_args()
    dev = load_split("dev", limit=args.limit)
    val = load_split("val", limit=args.limit)
    for r in dev + val:
        r["path"] = image_path(r)
    y = [r["label"] for r in val]

    def plain(matcher_cfg, variants):
        return dict(classifier=AlcoholClassifier(LexiconMatcher(matcher_cfg)), variants=variants,
                    extra_variants=(), caption_min_fraction=NO_CAPTION_RULE)

    steps = [
        ("original image only, exact words", plain(NO_RULES, ("original",))),
        ("+ mirror", plain(NO_RULES, ("original", "mirror"))),
        ("+ rotated ±35°", plain(NO_RULES, ("original", "mirror", "rot35", "rot-35"))),
        ("+ enlarged 2×", plain(NO_RULES, MAIN_VARIANTS)),
        ("+ leetspeak, negations, glued and split words", plain(NO_FUZZY, MAIN_VARIANTS)),
        ("+ approximate spelling", plain(MatcherConfig(), MAIN_VARIANTS)),
        ("+ caption-line rule", dict(extra_variants=())),
        ("+ extra variants when undecided (final)", {}),
    ]
    contributions = [metric_row(name, y, run(val, **kw)) for name, kw in steps]

    dev_text = [" ".join(texts(r, MAIN_VARIANTS).values()) for r in dev]
    val_text = [" ".join(texts(r, MAIN_VARIANTS).values()) for r in val]
    dev_y = [r["label"] for r in dev]
    models = [metric_row("rules (final system)", y, run(val))]
    for name, build in (("char n-gram alone", build_char_model), ("word n-gram alone", build_word_model)):
        models.append(metric_row(name, y, list(build().fit(dev_text, dev_y).predict(val_text))))

    grid = []
    for c in (0.5, 2.0, 8.0):
        model = build_char_model(c).fit(dev_text, dev_y)
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
            row = metric_row(f"rules + char n-gram fallback (C={c}, threshold={threshold})", y,
                             run(val, classifier=HybridClassifier(model, threshold)))
            grid.append(row)
    best = max(grid, key=lambda r: (round(r["f1"], 4), r["precision"]))
    models.append(best)

    write_report("components.md", f"Pipeline parts and trained models (validation split, n={len(val)})", [
        "## Adding one part at a time", table(contributions),
        "## Rules vs trained text classifiers",
        "The n-gram models are trained on the OCR text of the development split. "
        "The fallback row is the best setting from the grid below, chosen on validation "
        "itself, so it is optimistic. See heldout_brand.py for why it was not used.",
        table(models),
        "## Fallback settings", table(grid),
    ])


if __name__ == "__main__":
    main()
