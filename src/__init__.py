"""Alcohol-reference detection in superimposed image text (OCR + text rules).

* ``ocr`` reads the image (several variants) and returns text lines with boxes.
* ``normalize`` cleans one string. It does not know which words are drinks.
* ``lexicon`` holds the drink, brand, trap, and negation lists.
* ``classify`` turns cleaned text into a label, a confidence, and a reason.
* ``pipeline`` chooses which image variants to read and which line is the caption.
* ``metrics`` is used only by ``evaluate.py``.
"""
