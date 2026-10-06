"""Text normalization that undoes the dataset's adversarial text tricks.

The captions are deliberately obfuscated: random capitalisation
(``bourBoN``), leetspeak (``ch4mpa``, ``br3w``), glued words
(``Tryevent reServe``), split words (``smir noff``), and OCR confusions
(``l`` vs ``1`` vs ``i``). Each step here is independently switchable so its
contribution can be measured.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Digits/symbols that the generator (or OCR) substitutes for letters.
# A digit 1 is usually i (w1ne, g1n). On a long word it can also be l
# (absolut read as abso1ut). The l reading is a second exact candidate,
# not a replacement, and only for words of at least six letters.
LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    "@": "a", "$": "s", "!": "i", "|": "i", "€": "e",
})
LEET_MAP_L = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    "@": "a", "$": "s", "!": "i", "|": "i", "€": "e",
})
MIN_L_READING = 6

# Single characters OCR commonly confuses, mapped to one representative so
# that "whiskey" and an OCR misread "whlskey" produce the same skeleton.
# Multi-character confusions (rn -> m) are left to fuzzy matching, because
# they would break the character-to-token alignment the matcher relies on.
CONFUSABLE_MAP = str.maketrans({"l": "i", "j": "i", "0": "o", "q": "g", "v": "u"})


@dataclass(frozen=True)
class NormalizeConfig:
    """Switches for each normalization step (all on by default)."""

    strip_accents: bool = True
    leet: bool = True
    confusables: bool = True


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def basic_clean(text: str, cfg: NormalizeConfig = NormalizeConfig()) -> str:
    """Lowercase, strip accents, fix apostrophes, and squash whitespace."""
    if cfg.strip_accents:
        text = strip_accents(text)
    text = text.lower().replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def deleet_token(token: str) -> str:
    """Map leet characters to letters, only inside tokens that contain letters.

    Pure numbers (``2024``, ``0%``) are left alone so they keep their meaning.
    """
    if not re.search(r"[a-z]", token):
        return token
    return token.translate(LEET_MAP)


def _clean_token(token: str, cfg: NormalizeConfig) -> tuple[str, str]:
    """Primary reading, plus an l-for-1 reading when that reading is long enough."""
    token = token.strip("'")
    alt = ""
    if cfg.leet and re.search(r"[a-z]", token):
        primary = token.translate(LEET_MAP)
        if "1" in token:
            candidate = re.sub(r"[^a-z0-9]", "", token.translate(LEET_MAP_L))
            if len(candidate) >= MIN_L_READING and candidate != primary:
                alt = candidate
        token = primary
    token = re.sub(r"[^a-z0-9]", "", token)
    return token, alt if alt and alt != token else ""


def tokens(text: str, cfg: NormalizeConfig = NormalizeConfig()) -> list[str]:
    """Return normalized word tokens."""
    return [tok for tok, _ in token_pairs(text, cfg)]


def token_pairs(text: str, cfg: NormalizeConfig = NormalizeConfig()) -> list[tuple[str, str]]:
    """Tokens paired with an optional exact-only alternate (empty string if none)."""
    text = basic_clean(text, cfg)
    raw = re.findall(r"[a-z0-9@$!|€']+", text)
    out = []
    for tok in raw:
        cleaned, alt = _clean_token(tok, cfg)
        if cleaned:
            out.append((cleaned, alt))
    return out


def skeleton(text: str) -> str:
    """Collapse a string to a spacing- and confusion-invariant key.

    Used for substring matching of long terms so that glued (``bourbonevent``)
    and split (``smir noff``) words, and OCR letter confusions, still match.
    """
    return re.sub(r"[^a-z]", "", text).translate(CONFUSABLE_MAP)


@dataclass
class NormalizedText:
    raw: str
    tokens: list[str]
    joined: str  # tokens joined by single spaces, for phrase matching
    skeleton: str  # spaces removed + confusables folded
    # Same length as tokens. A non-empty entry is an exact-only second reading
    # (digit 1 as l). It is not fed to fuzzy, split, or skeleton matching.
    exact_aliases: list[str]


def normalize(text: str, cfg: NormalizeConfig = NormalizeConfig()) -> NormalizedText:
    pairs = token_pairs(text, cfg)
    toks = [tok for tok, _ in pairs]
    aliases = [alt for _, alt in pairs]
    joined = " ".join(toks)
    skel = skeleton("".join(toks)) if cfg.confusables else "".join(toks)
    return NormalizedText(raw=text, tokens=toks, joined=joined, skeleton=skel, exact_aliases=aliases)
