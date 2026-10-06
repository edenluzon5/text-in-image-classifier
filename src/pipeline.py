"""Image -> label: which image variants to read, and which text line is the caption.

Every main variant is always read. Stopping at the first alcohol word would
hide a negation that only a later variant reads (``beer`` on the original,
``alcohol-free beer`` after rotating). The extra variants run only when the
main ones find nothing that decides the image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .classify import AlcoholClassifier, Prediction
from .ocr import CachedOcr, OcrResult, frame_span, mean_confidence, polygon_long_side

MAIN_VARIANTS = ("original", "mirror", "rot35", "rot-35", "upscale2")
EXTRA_VARIANTS = (
    "rot180", "rot90", "rot-90", "mirror_rot35", "mirror_rot-35",
    "autocontrast", "upscale4", "sharpen",
)
# A caption drawn across the photo has a long edge; a sign inside the photo
# does not. 0.45 sits between the two on this data (a full-width caption is
# about 0.8 of the frame; the beer sign on 3044na is about 0.12).
CAPTION_MIN_FRACTION = 0.45


@dataclass
class PipelineOutput:
    prediction: Prediction
    texts: dict[str, str]
    variants_read: int


class AlcoholPipeline:
    def __init__(
        self,
        ocr: CachedOcr | None = None,
        classifier: AlcoholClassifier | None = None,
        variants=MAIN_VARIANTS,
        extra_variants=EXTRA_VARIANTS,
        min_line_confidence: float = 0.5,
        caption_min_fraction: float = CAPTION_MIN_FRACTION,
    ) -> None:
        self.ocr = ocr or CachedOcr()
        self.classifier = classifier or AlcoholClassifier()
        self.variants = tuple(variants)
        self.extra_variants = tuple(extra_variants)
        self.min_line_confidence = min_line_confidence
        self.caption_min_fraction = caption_min_fraction

    @classmethod
    def from_config(cls, path: str | Path, ocr: CachedOcr | None = None) -> "AlcoholPipeline":
        cfg = json.loads(Path(path).read_text())
        return cls(
            ocr=ocr,
            variants=cfg["main_variants"],
            extra_variants=cfg["extra_variants_if_undecided"],
            min_line_confidence=cfg["min_line_confidence"],
            caption_min_fraction=cfg["caption_min_width_fraction"],
        )

    def _caption_lines(self, result: OcrResult) -> list:
        """Drop short scene lines when a long caption line already decides.

        The long lines are used on their own only when they contain an alcohol
        term, a negation, or a known non-alcohol drink. A brand written on a
        short line is kept when the long line has no drink word in it.
        """
        kept = [d for d in result.detections
                if d.score >= self.min_line_confidence and d.text.strip()]
        if not kept:
            return kept
        span = frame_span(result)
        long = [d for d in kept if polygon_long_side(d.box) / span >= self.caption_min_fraction]
        if not long:
            return kept
        long_text = " ".join(d.text for d in long)
        evidence, negations, _ = self.classifier.lexicon_evidence({"line": long_text})
        drink = self.classifier.non_alcohol_drink({"line": long_text})
        return long if evidence or negations or drink else kept

    def _read(self, image_path: str, variants, texts, scores) -> None:
        for variant in variants:
            lines = self._caption_lines(self.ocr.read(image_path, variant))
            texts[variant] = " ".join(d.text for d in lines)
            scores[variant] = mean_confidence(lines)

    def _undecided(self, texts: dict[str, str], scores: dict[str, float]) -> bool:
        evidence, negations, _ = self.classifier.lexicon_evidence(texts, scores)
        return not (evidence or negations or self.classifier.non_alcohol_drink(texts))

    def run(self, image_path: str) -> PipelineOutput:
        texts: dict[str, str] = {}
        scores: dict[str, float] = {}
        self._read(image_path, self.variants, texts, scores)
        if self.extra_variants and self._undecided(texts, scores):
            self._read(image_path, self.extra_variants, texts, scores)
        return PipelineOutput(self.classifier.predict(texts, scores=scores), texts, len(texts))
