"""Trained text classifiers that were compared with the rules and not used.

Character n-grams (3-5 letters) survive broken and glued OCR words
(``smir noff``, ``bourbonevent``) better than whole words. They raise validation
F1 as a fallback, but they learn this dataset's caption template rather than the
drinks (see heldout_brand.py), so the final model does not use them.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import common  # noqa: F401  (puts the project root on sys.path)
from src.classify import ALCOHOL, AlcoholClassifier, Prediction
from src.normalize import normalize


def _normalize_batch(texts):
    return [normalize(t).joined for t in texts]


def build_char_model(c: float = 2.0) -> Pipeline:
    return Pipeline([
        ("normalize", FunctionTransformer(_normalize_batch)),
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
    ])


def build_word_model(c: float = 4.0) -> Pipeline:
    return Pipeline([
        ("normalize", FunctionTransformer(_normalize_batch)),
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
    ])


class HybridClassifier(AlcoholClassifier):
    """The rules first; the n-gram model decides only images the rules leave undecided."""

    def __init__(self, model: Pipeline, threshold: float, matcher=None) -> None:
        super().__init__(matcher)
        self.model, self.threshold = model, threshold
        self.alcohol_column = list(model.classes_).index(ALCOHOL)

    def predict(self, texts: dict[str, str], scores=None) -> Prediction:
        pred = super().predict(texts, scores)
        if pred.confidence != "low":
            return pred
        p = float(self.model.predict_proba([" ".join(texts.values())])[0][self.alcohol_column])
        if p >= self.threshold:
            return Prediction(ALCOHOL, "low", p, reason=f"char n-gram p={p:.2f}")
        return pred
