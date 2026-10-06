"""End-to-end checks of predict.py, run the way a grader runs it.

The images are rendered here, because data/images is not shipped. The OCR test
needs PaddleOCR and its models (downloaded on first use); set SKIP_OCR_TESTS=1
to run only the fast checks.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PREDICT = ROOT / "predict.py"
LABELS = {"alcohol", "non_alcohol"}


def render_caption(path: Path, caption: str) -> None:
    """A photo-like noisy background with a caption band, like the dataset."""
    img = Image.effect_noise((640, 427), 60).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 170, 640, 250), fill=(255, 255, 255))
    draw.text((20, 185), caption, fill=(20, 20, 160), font=ImageFont.load_default(size=40))
    img.save(path, quality=92)


def run_predict(input_dir: Path, output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PREDICT), "--input_dir", str(input_dir), "--output", str(output)],
        capture_output=True, text=True, timeout=900,
    )


def read_results(path: Path) -> tuple[list[str], dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        return header, {row[0]: row[1] for row in reader}


class PredictWithoutOcrTests(unittest.TestCase):
    def test_empty_directory_exits_with_error_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "results.csv")
            proc = run_predict(Path(tmp), out)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertFalse(out.exists())


@unittest.skipIf(os.environ.get("SKIP_OCR_TESTS") == "1", "SKIP_OCR_TESTS=1")
@unittest.skipIf(importlib.util.find_spec("paddleocr") is None, "paddleocr not installed")
class PredictEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        images = tmp / "images"
        (images / "nested").mkdir(parents=True)
        render_caption(images / "vodka.jpg", "Don't miss imported vodka event")
        render_caption(images / "nested" / "coffee.png", "Grab now hot coffee menu")
        (images / "broken.jpg").write_bytes(b"not an image")
        (images / "notes.txt").write_text("not an image either")
        cls.output = tmp / "out" / "results.csv"
        cls.proc = run_predict(images, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_runs_successfully(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-2000:])

    def test_csv_matches_the_example_format(self):
        header, _ = read_results(self.output)
        example_header, _ = read_results(ROOT / "data" / "example_results.csv")
        self.assertEqual(header, example_header)
        self.assertEqual(header, ["file_name", "prediction"])

    def test_one_row_per_image_including_subfolders_and_unreadable_files(self):
        _, rows = read_results(self.output)
        self.assertEqual(set(rows), {"vodka.jpg", "coffee.png", "broken.jpg"})
        self.assertTrue(set(rows.values()) <= LABELS)

    def test_an_unreadable_image_defaults_to_non_alcohol(self):
        _, rows = read_results(self.output)
        self.assertEqual(rows["broken.jpg"], "non_alcohol")

    def test_clear_captions_are_classified_correctly(self):
        _, rows = read_results(self.output)
        self.assertEqual(rows["vodka.jpg"], "alcohol")
        self.assertEqual(rows["coffee.png"], "non_alcohol")


if __name__ == "__main__":
    unittest.main()
