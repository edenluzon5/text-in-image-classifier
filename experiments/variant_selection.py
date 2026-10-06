"""Which image variants are worth reading? (REPORT.md section 3, "Image variants")

    python experiments/variant_selection.py

On the hand-transcribed images that name a drink, a variant "recovers" an image
when the classifier's matcher finds the caption's drink word in that variant's
OCR text. Starting from nothing, the variant that recovers the most new images
is added at each step, until no variant adds anything.
"""

from __future__ import annotations

import argparse
import statistics

from common import limit_arg, load_gold, ocr, table, write_report
from src.classify import LexiconMatcher

CANDIDATES = ["original", "mirror", "mirror_rot180", "rot20", "rot-20", "rot35", "rot-35",
              "mirror_rot20", "mirror_rot-20", "rot180"]


def main() -> None:
    args = limit_arg(argparse.ArgumentParser(description=__doc__)).parse_args()
    gold = [g for g in load_gold(args.limit) if g["keyword"]]
    hits = {v: set() for v in CANDIDATES}
    secs = {v: [] for v in CANDIDATES}
    for v in CANDIDATES:
        for g in gold:
            res = ocr(g, v)
            secs[v].append(res.seconds)
            if LexiconMatcher(terms=[g["keyword"]]).find(res.text())[0]:
                hits[v].add(g["file_name"])
    n = len(gold)

    alone = [{"variant": v, "recovered alone": len(hits[v]) / n,
              "seconds": statistics.mean(secs[v])} for v in CANDIDATES]
    steps, covered, remaining = [], set(), list(CANDIDATES)
    while remaining:
        best = max(remaining, key=lambda v: len(hits[v] - covered))
        gain = hits[best] - covered
        if not gain and steps:
            break
        covered |= gain
        remaining.remove(best)
        steps.append({"step": len(steps) + 1, "add variant": best, "newly recovered": len(gain),
                      "recovered so far": len(covered) / n,
                      "examples": ", ".join(sorted(gain)[:4])})
    never = sorted({g["file_name"] for g in gold} - covered)

    write_report("variant_selection.md", f"Greedy choice of image variants ({n} images with a drink word)", [
        "`mirror_rot180` is a vertical flip.",
        "## Each variant on its own", table(alone),
        "## Greedy selection", table(steps),
        f"Never recovered by any variant ({len(never)}): {', '.join(never) or 'none'}",
    ])


if __name__ == "__main__":
    main()
