"""What does each text rule add, and which tempting rules hurt? (REPORT.md section 5)

    python experiments/matching_ablation.py

On the development split (660 images), the rules-only decision is scored while the
text-cleaning and matching rules are switched on one at a time. Then two rejected
options are scored: approximate spelling on 5-letter words, and counting trap words
(*brew, bar, party*) as alcohol. A last table compares sets of image variants.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from common import limit_arg, load_split, metric_row, table, texts, write_report
from src import lexicon as lx
from src.classify import AlcoholClassifier, LexiconMatcher, MatcherConfig
from src.normalize import NormalizeConfig
from src.pipeline import MAIN_VARIANTS

VARIANTS = ("original", "mirror", "mirror_rot180")
BASE = MatcherConfig(normalize=NormalizeConfig(leet=False, confusables=False),
                     use_negations=False, use_skeleton=False, use_glue_split=False,
                     use_fuzzy=False)
LEET_ONLY = NormalizeConfig(confusables=False)
STEPS = [
    ("exact words (lower case only)", BASE),
    ("+ leetspeak", replace(BASE, normalize=LEET_ONLY)),
    ("+ negations", replace(BASE, normalize=LEET_ONLY, use_negations=True)),
    ("+ letters inside glued or split long words",
     replace(BASE, normalize=LEET_ONLY, use_negations=True, use_skeleton=True)),
    ("+ OCR letter mix-ups (l/i/j, 0/o, q/g, v/u)",
     replace(BASE, normalize=NormalizeConfig(), use_negations=True, use_skeleton=True)),
    ("+ template words peeled off, fragments rejoined", replace(MatcherConfig(), use_fuzzy=False)),
    ("+ approximate spelling, 6+ letters (final)", MatcherConfig()),
]
VARIANT_SETS = {
    "original, mirror, rot±35": ("original", "mirror", "rot35", "rot-35"),
    "  + vertical flip": ("original", "mirror", "mirror_rot180", "rot35", "rot-35"),
    "  + upscale2 (final main variants)": MAIN_VARIANTS,
}


def score(rows, classifier, variants) -> list[str]:
    return [classifier.predict(texts(r, variants, min_score=0.0)).label for r in rows]


def main() -> None:
    args = limit_arg(argparse.ArgumentParser(description=__doc__)).parse_args()
    dev = load_split("dev", limit=args.limit)
    y = [r["label"] for r in dev]

    steps = [metric_row(name, y, score(dev, AlcoholClassifier(LexiconMatcher(cfg)), VARIANTS),
                        "rules")
             for name, cfg in STEPS]
    rejected = [
        metric_row("approximate spelling on 5-letter words too", y,
                   score(dev, AlcoholClassifier(LexiconMatcher(MatcherConfig(fuzzy_min_len=5))),
                         VARIANTS), "rules"),
        metric_row("trap words count as alcohol", y,
                   score(dev, AlcoholClassifier(LexiconMatcher(terms=set(lx.ALL_TERMS) | lx.TRAPS)),
                         VARIANTS), "rules"),
    ]
    variant_rows = [metric_row(name, y, score(dev, AlcoholClassifier(), vs), "variants read")
                    for name, vs in VARIANT_SETS.items()]

    write_report("matching_ablation.md", f"Text rules, one at a time (development split, n={len(dev)})", [
        "Rules only, all OCR lines kept, variants read: original, mirror, vertical flip.",
        "## Adding one rule at a time", table(steps),
        "## Rejected options (each on top of the final rules)", table(rejected),
        "## Which image variants to read (final rules)", table(variant_rows),
    ])


if __name__ == "__main__":
    main()
