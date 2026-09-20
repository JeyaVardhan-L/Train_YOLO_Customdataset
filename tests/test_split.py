"""Tests for dataset splitting functionality."""

import unittest
import tempfile
import shutil
from pathlib import Path

from src.data.split import split_dataset


class TestSplitDataset(unittest.TestCase):
    """Test splitting behavior on mock temporary dataset."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "raw"
        self.output_dir = Path(self.temp_dir) / "split"
        self.input_dir.mkdir()

        # Create 10 mock image and label pairs
        for i in range(10):
            img_file = self.input_dir / f"img_{i:02d}.jpg"
            lbl_file = self.input_dir / f"img_{i:02d}.txt"
            img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)  # Mock JPEG header
            lbl_file.write_text(f"2 0.5 0.5 0.2 0.2\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_split_ratio(self):
        train_count, val_count = split_dataset(
            input_path=str(self.input_dir),
            output_path=str(self.output_dir),
            train_pct=0.8,
            seed=42,
            move_files=False,
        )
        self.assertEqual(train_count, 8)
        self.assertEqual(val_count, 2)

        # Check directory existence
        train_imgs = list((self.output_dir / "train" / "images").glob("*.jpg"))
        train_lbls = list((self.output_dir / "train" / "labels").glob("*.txt"))
        val_imgs = list((self.output_dir / "validation" / "images").glob("*.jpg"))
        val_lbls = list((self.output_dir / "validation" / "labels").glob("*.txt"))

        self.assertEqual(len(train_imgs), 8)
        self.assertEqual(len(train_lbls), 8)
        self.assertEqual(len(val_imgs), 2)
        self.assertEqual(len(val_lbls), 2)

    def test_invalid_train_pct(self):
        with self.assertRaises(ValueError):
            split_dataset(str(self.input_dir), str(self.output_dir), train_pct=1.5)


if __name__ == "__main__":
    unittest.main()
