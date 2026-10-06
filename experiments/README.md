# Experiments

The development experiments behind REPORT.md sections 3–5. They are separate from
the final model (`src/`): nothing here is needed to run `predict.py`.

| script | question | result recorded during development | report |
|---|---|---|---|
| `ocr_comparison.py` | Which OCR engine reads these captions best? | drink word found: PaddleOCR 0.74, EasyOCR 0.52, Tesseract 0.16 | 3 |
| `variant_selection.py` | Which image variants are worth reading? | original, mirror, rotated +35°, upside down, rotated −35° recover 88% of the drink words; upside down was then moved to the extra variants (one more hit and one more false alarm on development) | 3 |
| `matching_ablation.py` | What does each text rule add, and which tempting rules hurt? | trap words as evidence: precision 0.986 → 0.595; approximate spelling on 5-letter words: +2 hits, +3 false alarms | 5 |
| `components.py` | What does each pipeline part add? How do trained text classifiers compare? | F1 0.667 (original image, exact words) → 0.948 (all rules); word n-gram alone 0.876; char n-gram fallback 0.968 | 4, 5 |
| `heldout_brand.py` | What happens with a brand the model has never seen? | brand removed: rules find 0–6% of its images, rules + n-gram 24–78%, but with 2.7% false alarms | 5 |
| `synthetic_captions.py` | Does it work on drinks absent from the dataset, and on negations? | every unseen drink found; 119 of 120 correct after the negation fix | 5 |
| `augmentation_stress.py` | Does it survive rotation, blur, occlusion, perspective…? | precision 1.000, recall 0.865 on 104 transformed images | 5 |

The results above were recorded with earlier versions of the code while it was
being developed. The scripts now run against the final code, so a rerun can give
slightly different numbers (for example, the final pipeline also has the
caption-line rule, and the augmentation script picks its photos with a different
random generator).

## Running

```bash
pip install -r experiments/requirements.txt    # adds scikit-learn; EasyOCR, PyTorch, pytesseract for ocr_comparison.py
brew install tesseract                         # ocr_comparison.py only (Linux: apt install tesseract-ocr)
python experiments/components.py               # each script writes results/experiments/<name>.md
```

The labelled images must be in `data/images/` (`python download_data.py`).
PaddleOCR output is cached in `cache/`, shared with `evaluate.py`. On a laptop CPU
the first run of `matching_ablation.py` or `components.py` reads the 660
development images (about an hour). The other scripts take 10–20 minutes, and
`heldout_brand.py` takes seconds once `components.py` has run.
For a quick check that a script runs, add `--limit 6` (or `--per_cell 1` and
`--per_class 1` for the two image-generating scripts).

| file | role |
|---|---|
| `common.py` | data splits, cached OCR, metrics, markdown output |
| `ngram.py` | the character and word n-gram classifiers that were compared and not used |
