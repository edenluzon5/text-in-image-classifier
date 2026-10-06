"""Decide alcohol / non_alcohol from OCR text, with evidence and confidence.

Decision order
--------------
1. Negation phrases (``ginger ale``, ``alcohol-free``) are removed from the
   text of each image variant. A drink word cancelled in any variant is
   dropped from every variant, so a bare ``beer`` in one orientation cannot
   outvote ``alcohol-free beer`` in another.
2. The lexicon is matched in increasingly permissive ways:
   ``exact`` token -> ``phrase`` -> ``glued`` (template words peeled off) ->
   ``split`` -> ``skeleton`` substring -> ``fuzzy`` (bounded edit distance).
   Ambiguous terms (``porter``, ``corona``, ``cider``) stop at exact, phrase,
   and glued. Split, skeleton, and fuzzy hits count only on the image variant
   whose OCR confidence is highest, when per-variant scores are provided.
3. If nothing matches, the image is non-alcohol.

Confidence
----------
* ``high``   - exact or phrase match on an unambiguous term
* ``medium`` - glued, split, skeleton or fuzzy match, an ambiguous term, a
  negation, or a clearly read non-alcohol drink
* ``low``    - no evidence either way; predicted non-alcohol
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from . import lexicon as lx
from .normalize import NormalizeConfig, normalize, skeleton, tokens

ALCOHOL = "alcohol"
NON_ALCOHOL = "non_alcohol"
METHOD_RANK = {"exact": 0, "phrase": 0, "glued": 1, "split": 1, "skeleton": 1, "fuzzy": 2}
# Ambiguous terms have a common non-alcohol sense, so only a literal token,
# a literal phrase, or a template word peeled off the token may match them.
# Fuzzy and skeleton matches are how "potter" became "porter".
AMBIGUOUS_METHODS = {"exact", "phrase", "glued"}
# Split, skeleton, and fuzzy reads are easy to invent from a mirrored or
# rotated variant. They count only for the variant with the highest OCR confidence.
BEST_VARIANT_METHODS = {"split", "skeleton", "fuzzy"}


def peel_template_words(token: str, min_rest: int = 3) -> list[str]:
    """Split ad-template words off the start and end of a glued token.

    ``importedrum`` -> ``["imported", "rum"]``;
    ``foreventpremiumjhnniewalker`` -> ``["for", "event", "premium", "jhnniewalker"]``.
    Returns ``[]`` if nothing was peeled.
    """
    pieces_front, pieces_back, rest = [], [], token
    changed = True
    while changed:
        changed = False
        for w in lx.TEMPLATE_WORDS:
            if rest.startswith(w) and len(rest) - len(w) >= min_rest:
                pieces_front.append(w)
                rest = rest[len(w):]
                changed = True
                break
            if rest.endswith(w) and len(rest) - len(w) >= min_rest:
                pieces_back.insert(0, w)
                rest = rest[: -len(w)]
                changed = True
                break
    if not pieces_front and not pieces_back:
        return []
    return pieces_front + [rest] + pieces_back


@dataclass(frozen=True)
class MatcherConfig:
    """Switches for each matching stage (used by the ablation experiments)."""

    normalize: NormalizeConfig = NormalizeConfig()
    use_negations: bool = True
    use_skeleton: bool = True
    use_glue_split: bool = True
    use_fuzzy: bool = True
    skeleton_min_len: int = 5
    fuzzy_min_len: int = 6


@dataclass
class Match:
    term: str
    method: str
    matched_text: str
    variant: str = "original"

    @property
    def ambiguous(self) -> bool:
        return self.term in lx.AMBIGUOUS

    @property
    def strength(self) -> str:
        if self.method in ("exact", "phrase") and not self.ambiguous:
            return "high"
        return "medium"


@dataclass
class Prediction:
    label: str
    confidence: str
    score: float  # probability-like score for alcohol, used for PR curves
    evidence: list[Match] = field(default_factory=list)
    negations: list[str] = field(default_factory=list)
    reason: str = ""


class LexiconMatcher:
    def __init__(self, cfg: MatcherConfig = MatcherConfig(), terms=None) -> None:
        self.cfg = cfg
        terms = set(terms if terms is not None else lx.ALL_TERMS)
        self.single = {t for t in terms if " " not in t}
        self.phrases = sorted((t for t in terms if " " in t), key=len, reverse=True)
        self.skeletons = {
            t: skeleton(t.replace(" ", ""))
            for t in terms
            if len(t.replace(" ", "")) >= cfg.skeleton_min_len
        }
        # Fuzzy matching compares confusion-folded skeletons; multi-word terms
        # are included with spaces removed ("johnniewalker").
        self.fuzzy_keys: dict[str, str] = {}
        for t in terms:
            key = skeleton(t.replace(" ", ""))
            if len(key) >= cfg.fuzzy_min_len:
                self.fuzzy_keys.setdefault(key, t)
        self.fuzzy_list = list(self.fuzzy_keys)
        # Negations are compared with spaces removed, so OCR-split forms
        # ("alco hol-free", "non-alc oholic") are still recognised.
        self.neg_modifiers = {"".join(tokens(n)) for n in lx.NEGATION_MODIFIERS}
        self.neg_phrases = {"".join(tokens(n)) for n in lx.NEGATIONS}
        self.protected = lx.FUZZY_BLOCKLIST | lx.TRAPS | self.neg_phrases | {
            w for w in lx.NON_ALCOHOL_DRINKS if " " not in w
        }

    def _terms_in_span(self, span: list[str]) -> set[str]:
        """Alcohol terms written inside a negated span."""
        found = {t for t in span if t in self.single}
        joined = f" {' '.join(span)} "
        found.update(p for p in self.phrases if f" {p} " in joined)
        return found

    def _span_terms(self, span: list[str], aliases: list[str]) -> set[str]:
        found = self._terms_in_span(span)
        found.update(a for a in aliases if a in self.single)
        return found

    def _strip_negations(
        self, toks: list[str], aliases: list[str] | None = None,
    ) -> tuple[list[str], list[str], set[str], list[str]]:
        """Remove negated spans.

        Returns remaining tokens, the negation phrases, alcohol terms that
        the negation cancels, and the exact-only aliases of the tokens that
        remain. Numeric-only modifiers such as ``00`` (from ``0.0``) still
        strip their own span, but they do not veto other variants: that form
        is a common OCR accident.
        """
        if aliases is None:
            aliases = [""] * len(toks)
        keep, kept_aliases, negated, vetoed, i = [], [], [], set(), 0
        while i < len(toks):
            for k in (4, 3, 2, 1):
                glued = "".join(toks[i : i + k])
                if glued in self.neg_modifiers:
                    end = i + k + lx.NEGATION_SCOPE
                    span = toks[i:end]
                    negated.append(" ".join(span))
                    if re.search(r"[a-z]", glued):
                        vetoed |= self._span_terms(span, aliases[i:end])
                    i = end
                    break
                if glued in self.neg_phrases:
                    end = i + k
                    span = toks[i:end]
                    negated.append(" ".join(span))
                    if re.search(r"[a-z]", glued):
                        vetoed |= self._span_terms(span, aliases[i:end])
                    i = end
                    break
            else:
                keep.append(toks[i])
                kept_aliases.append(aliases[i])
                i += 1
        return keep, negated, vetoed, kept_aliases

    def find(self, text: str, variant: str = "original") -> tuple[list[Match], list[str], set[str]]:
        norm = normalize(text, self.cfg.normalize)
        toks, aliases, negated, vetoed = list(norm.tokens), list(norm.exact_aliases), [], set()
        if self.cfg.use_negations:
            toks, negated, vetoed, aliases = self._strip_negations(toks, aliases)
        matches: dict[str, Match] = {}

        def add(term: str, method: str, matched: str) -> None:
            if term in lx.AMBIGUOUS and method not in AMBIGUOUS_METHODS:
                return
            prev = matches.get(term)
            if prev is None or METHOD_RANK[method] < METHOD_RANK[prev.method]:
                matches[term] = Match(term, method, matched, variant)

        for tok in toks:
            if tok in self.single:
                add(tok, "exact", tok)
        for alias in aliases:
            if alias and alias in self.single:
                add(alias, "exact", alias)
        for phrase in self.phrases:
            if f" {phrase} " in f" {' '.join(toks)} ":
                add(phrase, "phrase", phrase)

        extra: list[str] = []
        if self.cfg.use_glue_split:
            for tok in toks:
                pieces = peel_template_words(tok) if len(tok) >= 6 else []
                for piece in pieces:
                    if piece in self.single:
                        add(piece, "glued", tok)
                extra += pieces
            # "g in" -> gin, "win e" -> wine: rejoin fragments of a split word
            for a, b in zip(toks, toks[1:]):
                if min(len(a), len(b)) <= 2 and a + b in self.single:
                    add(a + b, "split", f"{a} {b}")

        if self.cfg.use_skeleton:
            self._skeleton_matches(toks, add)
        if self.cfg.use_fuzzy:
            self._fuzzy_matches(toks, extra, add)
        return list(matches.values()), negated, vetoed

    def _skeleton_matches(self, toks: list[str], add) -> None:
        tok_skels = [skeleton(t) if self.cfg.normalize.confusables else t for t in toks]
        starts, pos = [], 0
        for s in tok_skels:
            starts.append(pos)
            pos += len(s)
        full = "".join(tok_skels)
        for term, skel in self.skeletons.items():
            for m in re.finditer(re.escape(skel), full):
                first = max(i for i, st in enumerate(starts) if st <= m.start())
                last = max(i for i, st in enumerate(starts) if st < m.end())
                covered = toks[first : last + 1]
                if first == last and covered[0] in lx.SUBSTRING_BLOCKLIST:
                    continue
                add(term, "skeleton", " ".join(covered))
                break

    def _fuzzy_matches(self, toks: list[str], extra: list[str], add) -> None:
        candidates = (
            list(toks) + extra
            + [a + b for a, b in zip(toks, toks[1:])]
            + [a + b + c for a, b, c in zip(toks, toks[1:], toks[2:])]
        )
        for cand in candidates:
            if cand in self.protected:
                continue
            key = skeleton(cand)
            if len(key) < self.cfg.fuzzy_min_len:
                continue
            max_dist = 1 if len(key) < 9 else 2
            hit = process.extractOne(
                key, self.fuzzy_list, scorer=Levenshtein.distance, score_cutoff=max_dist
            )
            if hit is not None:
                add(self.fuzzy_keys[hit[0]], "fuzzy", cand)


class AlcoholClassifier:
    """Combines lexicon evidence across the image variants."""

    def __init__(self, matcher: LexiconMatcher | None = None) -> None:
        self.matcher = matcher or LexiconMatcher()

    def non_alcohol_drink(self, texts: dict[str, str]) -> str | None:
        """First known non-alcohol drink read exactly in any variant, if any."""
        for text in texts.values():
            joined = f" {normalize(text, self.matcher.cfg.normalize).joined} "
            for drink in lx.NON_ALCOHOL_DRINKS:
                if f" {drink} " in joined:
                    return drink
        return None

    def lexicon_evidence(self, texts: dict[str, str], scores: dict[str, float] | None = None):
        evidence, negations, vetoed = [], [], set()
        for variant, text in texts.items():
            found, neg, span_veto = self.matcher.find(text, variant)
            evidence += found
            negations += neg
            vetoed |= span_veto
        evidence = [m for m in evidence if m.term not in vetoed]
        evidence = self._weak_from_best_variant(evidence, scores)
        return evidence, negations, vetoed

    def _weak_from_best_variant(self, evidence: list[Match], scores: dict[str, float] | None):
        """Keep short skeleton / fuzzy / split hits only from the clearest variant.

        A drink or brand of 7 letters or more still counts from any variant,
        except generic words such as ``brewery``. Those long hits were being
        dropped when a sign in the photo outscored the caption (``jack daniels``
        under ``SHORTBUS``). Short hits, and generic words, stay on the clearest
        variant, which is what blocks mirror garbage such as ``grappa``. Exact,
        phrase, and glued matches count from every variant. With no scores, or
        with every variant at zero confidence, nothing is dropped.
        """
        if not scores or not evidence:
            return evidence
        best = max(scores.values())
        if best <= 0:
            return evidence
        best_variant = next(variant for variant, score in scores.items() if score == best)
        return [
            m for m in evidence
            if m.method not in BEST_VARIANT_METHODS
            or m.variant == best_variant
            or (
                len(m.term.replace(" ", "")) >= 7 and m.term not in lx.GENERIC
            )
        ]

    def predict(self, texts: dict[str, str], scores: dict[str, float] | None = None) -> Prediction:
        """``texts`` maps variant name -> OCR text for that variant.

        ``scores`` maps the same names to mean OCR confidence. Weak matches are
        then taken only from the highest-scoring variant.
        """
        evidence, negations, _vetoed = self.lexicon_evidence(texts, scores)
        if evidence:
            best = min(evidence, key=lambda m: (m.strength != "high", METHOD_RANK[m.method]))
            conf = best.strength
            score = 0.97 if conf == "high" else 0.85
            return Prediction(ALCOHOL, conf, score, evidence, negations,
                              f"{best.method} match '{best.term}' ({best.variant})")
        if negations:
            return Prediction(NON_ALCOHOL, "medium", 0.1, [], negations,
                              f"negation: {negations[0]}")
        drink = self.non_alcohol_drink(texts)
        if drink:
            return Prediction(NON_ALCOHOL, "medium", 0.05, [], [],
                              f"non-alcohol drink '{drink}', no alcohol term")
        return Prediction(NON_ALCOHOL, "low", 0.0, [], [], "no alcohol term found")
