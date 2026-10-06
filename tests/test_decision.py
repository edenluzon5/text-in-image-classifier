"""Decision rules that have to hold on a dataset this lexicon has not seen."""

from __future__ import annotations

import unittest

from src.classify import ALCOHOL, NON_ALCOHOL, AlcoholClassifier
from src.ocr import Detection, OcrResult
from src.pipeline import AlcoholPipeline


def decide(texts, scores=None):
    return AlcoholClassifier().predict(texts, scores=scores)


class NegationTests(unittest.TestCase):
    def test_negation_in_one_variant_cancels_a_bare_drink_in_another(self):
        pred = decide({
            "original": "cold beer tonight",
            "mirror": "grab now alcohol-free beer",
        })
        self.assertEqual(pred.label, NON_ALCOHOL)
        self.assertIn("negation", pred.reason)

    def test_a_different_brand_survives_an_unrelated_negation(self):
        pred = decide({"original": "try heineken tonight", "mirror": "ginger ale"})
        self.assertEqual(pred.label, ALCOHOL)
        self.assertIn("heineken", pred.reason)

    def test_ginger_ale_in_a_single_variant_is_not_alcohol(self):
        pred = decide({"original": "don't miss ginger ale event"})
        self.assertEqual(pred.label, NON_ALCOHOL)

    def test_numeric_ocr_accident_does_not_veto_another_variant(self):
        # "0.0" normalizes to "00", which shows up in garbage OCR.
        pred = decide({"original": "cold beer", "mirror": "00 akv wn"})
        self.assertEqual(pred.label, ALCOHOL)
        self.assertIn("beer", pred.reason)


class SynonymTests(unittest.TestCase):
    def test_alternate_brand_spellings_match(self):
        for text in ("try gray goose", "johnny walker night", "cold maccabi", "gold star tonight"):
            pred = decide({"original": text})
            self.assertEqual(pred.label, ALCOHOL, text)

    def test_jager_is_jagermeister(self):
        pred = decide({"original": "get jager tonight"})
        self.assertEqual(pred.label, ALCOHOL)
        self.assertIn("jager", pred.reason)

    def test_short_nicknames_are_not_brands(self):
        for text in ("meet bud after work", "ask jd", "henny from accounting"):
            pred = decide({"original": text})
            self.assertEqual(pred.label, NON_ALCOHOL, text)


class AmbiguousTests(unittest.TestCase):
    def test_potter_does_not_fuzzy_match_porter(self):
        pred = decide({"original": "HYARY POTTER and the deathly hallows"})
        self.assertEqual(pred.label, NON_ALCOHOL)

    def test_exact_ambiguous_term_still_matches(self):
        pred = decide({"original": "try a porter tonight"})
        self.assertEqual(pred.label, ALCOHOL)
        self.assertEqual(pred.confidence, "medium")

    def test_glued_ambiguous_term_still_matches(self):
        pred = decide({"original": "PORTLIMITED edition"})
        self.assertEqual(pred.label, ALCOHOL)
        self.assertIn("port", pred.reason)

    def test_split_ambiguous_term_does_not_match(self):
        pred = decide({"original": "po rter special"})
        self.assertEqual(pred.label, NON_ALCOHOL)


class BestVariantTests(unittest.TestCase):
    def test_mirror_garbage_loses_to_a_clearer_variant(self):
        pred = decide(
            {"original": "Grab now weekend discount", "mirror": "Grap p a"},
            {"original": 0.93, "mirror": 0.41},
        )
        self.assertEqual(pred.label, NON_ALCOHOL)

    def test_weak_match_on_the_clearest_variant_counts(self):
        pred = decide(
            {"original": "weekend special", "mirror": "vod ka"},
            {"original": 0.30, "mirror": 0.91},
        )
        self.assertEqual(pred.label, ALCOHOL)
        self.assertIn("vodka", pred.reason)

    def test_weak_match_on_a_worse_variant_is_dropped(self):
        pred = decide(
            {"original": "weekend special offer", "mirror": "vod ka"},
            {"original": 0.90, "mirror": 0.30},
        )
        self.assertEqual(pred.label, NON_ALCOHOL)

    def test_a_long_brand_survives_a_clearer_but_empty_variant(self):
        pred = decide(
            {"original": "SHORTBUS", "upscale2": "premiumjack daniels event"},
            {"original": 0.98, "upscale2": 0.90},
        )
        self.assertEqual(pred.label, ALCOHOL)
        self.assertIn("jack daniel", pred.reason)

    def test_a_generic_word_on_a_worse_variant_does_not_count(self):
        pred = decide(
            {"original": "hot chocolate tonight", "rot-35": "brewerx"},
            {"original": 0.95, "rot-35": 0.40},
        )
        self.assertEqual(pred.label, NON_ALCOHOL)

    def test_exact_match_counts_from_any_variant(self):
        pred = decide(
            {"original": "hello there", "mirror": "wine"},
            {"original": 0.95, "mirror": 0.40},
        )
        self.assertEqual(pred.label, ALCOHOL)
        self.assertEqual(pred.confidence, "high")


class _ScriptedOcr:
    def __init__(self, pages: dict[str, tuple[str, float]]):
        self.pages = pages
        self.calls: list[str] = []

    def read(self, image_path: str, variant: str) -> OcrResult:
        self.calls.append(variant)
        text, score = self.pages.get(variant, ("", 0.0))
        dets = [Detection(text, score, [[0, 0], [1, 0], [1, 1], [0, 1]])] if text else []
        return OcrResult(variant, 8, 8, dets, seconds=0.0)


class PipelineTests(unittest.TestCase):
    def test_all_variants_run_so_a_later_negation_is_seen(self):
        ocr = _ScriptedOcr({
            "original": ("cold beer", 0.99),
            "mirror": ("alcohol-free beer", 0.90),
        })
        pipe = AlcoholPipeline(ocr=ocr, variants=("original", "mirror"), extra_variants=())
        out = pipe.run("ignored.jpg")
        self.assertEqual(ocr.calls, ["original", "mirror"])
        self.assertEqual(out.prediction.label, NON_ALCOHOL)

    def test_config_loads(self):
        pipe = AlcoholPipeline.from_config("config.json")
        self.assertEqual(pipe.variants, ("original", "mirror", "rot35", "rot-35", "upscale2"))
        for extra in ("rot180", "mirror_rot35", "upscale4", "sharpen"):
            self.assertIn(extra, pipe.extra_variants)
        self.assertEqual(pipe.min_line_confidence, 0.5)

    def test_extra_variants_run_only_when_nothing_was_read(self):
        ocr = _ScriptedOcr({
            "original": ("", 0.0),
            "rot180": ("upside down whiskey", 0.9),
        })
        pipe = AlcoholPipeline(
            ocr=ocr, variants=("original",), extra_variants=("rot180", "rot90"),
        )
        out = pipe.run("ignored.jpg")
        self.assertEqual(ocr.calls, ["original", "rot180", "rot90"])
        self.assertEqual(out.prediction.label, ALCOHOL)
        self.assertIn("whiskey", out.prediction.reason)

    def test_extra_variants_do_not_run_after_a_clear_soft_drink(self):
        ocr = _ScriptedOcr({
            "original": ("orange juice", 0.95),
            "rot180": ("beer", 0.99),
        })
        pipe = AlcoholPipeline(ocr=ocr, variants=("original",), extra_variants=("rot180",))
        out = pipe.run("ignored.jpg")
        self.assertEqual(ocr.calls, ["original"])
        self.assertEqual(out.prediction.label, NON_ALCOHOL)


class DigitOneReadingTests(unittest.TestCase):
    def test_long_word_reads_one_as_l(self):
        # absolut is ambiguous, so fuzzy is blocked. The l reading is exact.
        pred = decide({"original": "smooth abso1ut explore"})
        self.assertEqual(pred.label, ALCOHOL)
        self.assertIn("absolut", pred.reason)

    def test_short_ambiguous_word_does_not_get_the_l_reading(self):
        # stoli is ambiguous and only five letters, so neither skeleton nor
        # the l-for-1 reading may turn "sto1i" into it.
        pred = decide({"original": "sto1i tonight"})
        self.assertEqual(pred.label, NON_ALCOHOL)

    def test_one_still_reads_as_i(self):
        pred = decide({"original": "g1n night"})
        self.assertEqual(pred.label, ALCOHOL)
        self.assertIn("gin", pred.reason)

    def test_negation_cancels_the_l_reading(self):
        pred = decide({"original": "alcohol-free abso1ut"})
        self.assertEqual(pred.label, NON_ALCOHOL)


class PrevalenceTests(unittest.TestCase):
    def test_precision_falls_when_alcohol_is_rare(self):
        from src.metrics import precision_at_prevalence
        balanced = precision_at_prevalence(recall=0.92, fpr=0.01, prevalence=0.5)
        rare = precision_at_prevalence(recall=0.92, fpr=0.01, prevalence=0.02)
        self.assertGreater(balanced, 0.95)
        self.assertLess(rare, balanced)
        self.assertGreater(rare, 0.5)


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class _BoxedOcr:
    def __init__(self, pages: dict[str, list]):
        self.pages = pages

    def read(self, image_path: str, variant: str) -> OcrResult:
        return OcrResult(variant, 640, 480, self.pages.get(variant, []), seconds=0.0)


class OverlayTests(unittest.TestCase):
    def test_a_long_caption_outranks_a_sign_in_the_photo(self):
        ocr = _BoxedOcr({
            "original": [
                Detection("MUSIC COLD BEER", 0.92, _box(438, 67, 514, 89)),
                Detection(
                    "Looking for brew sparkling water check", 0.98, _box(99, 313, 628, 346),
                ),
            ],
        })
        pipe = AlcoholPipeline(ocr=ocr, variants=("original",), extra_variants=())
        out = pipe.run("3044na.jpg")
        self.assertEqual(out.prediction.label, NON_ALCOHOL)
        self.assertNotIn("beer", out.prediction.reason)

    def test_a_short_brand_is_kept_when_the_long_line_names_no_drink(self):
        ocr = _BoxedOcr({
            "original": [
                Detection("Looking for premium event", 0.90, _box(20, 300, 560, 340)),
                Detection("tequila", 0.95, _box(40, 40, 130, 70)),
            ],
        })
        pipe = AlcoholPipeline(ocr=ocr, variants=("original",), extra_variants=())
        out = pipe.run("brand.jpg")
        self.assertEqual(out.prediction.label, ALCOHOL)
        self.assertIn("tequila", out.prediction.reason)

    def test_alcohol_on_the_long_line_still_counts(self):
        ocr = _BoxedOcr({
            "original": [
                Detection("cold beer tonight", 0.99, _box(20, 300, 560, 340)),
                Detection("SALOON", 0.99, _box(40, 40, 140, 70)),
            ],
        })
        pipe = AlcoholPipeline(ocr=ocr, variants=("original",), extra_variants=())
        out = pipe.run("beer.jpg")
        self.assertEqual(out.prediction.label, ALCOHOL)
        self.assertIn("beer", out.prediction.reason)


class VariantTests(unittest.TestCase):
    def test_variants_keep_or_grow_the_frame(self):
        from PIL import Image
        from src.ocr import make_variant
        image = Image.new("RGB", (80, 40), (240, 240, 240))
        self.assertEqual(make_variant(image, "sharpen").size, image.size)
        self.assertEqual(make_variant(image, "mirror").size, image.size)
        self.assertEqual(make_variant(image, "upscale2").size, (160, 80))
        self.assertGreater(make_variant(image, "mirror_rot35").size[1], 40)

    def test_unknown_variant_is_an_error(self):
        from PIL import Image
        from src.ocr import make_variant
        with self.assertRaises(ValueError):
            make_variant(Image.new("RGB", (8, 8)), "hflip")


if __name__ == "__main__":
    unittest.main()
