"""What happens with a brand the model has never seen? (REPORT.md section 5)

    python experiments/heldout_brand.py

A random split puts every brand in development, validation and test, so the test
score says nothing about new brands. Here each frequent brand is removed in turn:
from the rules' brand list, and from the n-gram model's training images. Recall is
measured on the development + validation images whose caption names that brand,
and the n-gram's false-alarm rate on validation non-alcohol images.
"""

from __future__ import annotations

import argparse

from common import limit_arg, load_split, table, texts, write_report
from ngram import HybridClassifier, build_char_model
from src import lexicon as lx
from src.classify import ALCOHOL, AlcoholClassifier, LexiconMatcher
from src.pipeline import MAIN_VARIANTS

BRANDS = ["smirnoff", "bacardi", "jack daniels", "heineken", "guinness", "budweiser",
          "johnnie walker", "stella artois", "absolut"]
NGRAM_C, NGRAM_THRESHOLD = 2.0, 0.6


def main() -> None:
    args = limit_arg(argparse.ArgumentParser(description=__doc__)).parse_args()
    pool = load_split("dev", "val", limit=args.limit)
    pool_texts = [texts(r, MAIN_VARIANTS) for r in pool]
    is_dev = [r["split"] == "dev" for r in pool]
    negatives = [i for i, r in enumerate(pool) if r["label"] != ALCOHOL and not is_dev[i]]
    full = AlcoholClassifier()
    evidence = [{m.term for m in full.lexicon_evidence(t)[0]} for t in pool_texts]

    rows = []
    for brand in BRANDS:
        idx = [i for i, ev in enumerate(evidence) if brand in ev]
        if not idx:
            continue
        terms = {t for t in lx.ALL_TERMS if t != brand and not t.startswith(brand.split()[0])}
        rules = AlcoholClassifier(LexiconMatcher(terms=terms))
        train = [i for i in range(len(pool)) if is_dev[i] and i not in set(idx)]
        model = build_char_model(NGRAM_C).fit([" ".join(pool_texts[i].values()) for i in train],
                                              [pool[i]["label"] for i in train])
        hybrid = HybridClassifier(model, NGRAM_THRESHOLD, LexiconMatcher(terms=terms))
        found = lambda clf, ids: sum(clf.predict(pool_texts[i]).label == ALCOHOL for i in ids)  # noqa: E731
        rows.append({
            "brand removed": brand, "images": len(idx),
            "recall, rules": found(rules, idx) / len(idx),
            "recall, rules + n-gram": found(hybrid, idx) / len(idx),
            "n-gram false alarms": found(hybrid, negatives) / max(1, len(negatives)),
        })

    write_report("heldout_brand.md", "Remove-one-brand test (development + validation images)", [
        "A rule list cannot know a brand it does not contain. The n-gram model finds more "
        "of these images, but by recognising the caption template rather than the drink, "
        "which shows up as false alarms on non-alcohol captions with the same template.",
        table(rows),
    ])


if __name__ == "__main__":
    main()
