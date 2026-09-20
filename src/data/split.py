"""Dataset splitting logic to create train and validation splits from raw pairs."""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def split_dataset(
    input_path: str,
    output_path: str = "data",
    train_pct: float = 0.8,
    seed: int = 42,
    move_files: bool = False
) -> Tuple[int, int]:
    """Split paired images and YOLO label files into train and validation directories.

    Expected input structure can be either:
      1) input_path/images and input_path/labels
      2) input_path/ with flat mixed images and .txt label files

    Output structure created:
      output_path/
      ├── train/
      │   ├── images/
      │   └── labels/
      └── validation/
          ├── images/
          └── labels/

    Args:
        input_path: Root of input data.
        output_path: Directory to create train/val sets in.
        train_pct: Proportion of data allocated to train set (e.g. 0.8).
        seed: Random seed for deterministic reproducibility.
        move_files: If True moves files instead of copying.

    Returns:
        tuple: (train_count, val_count)
    """
    input_dir = Path(input_path).resolve()
    out_dir = Path(output_path).resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not (0.01 <= train_pct <= 0.99):
        raise ValueError(f"train_pct must be between 0.01 and 0.99, got {train_pct}")

    # Check for subfolder structure vs flat directory
    sub_images = input_dir / "images"
    sub_labels = input_dir / "labels"

    if sub_images.is_dir():
        image_files = [p for p in sub_images.iterdir() if p.is_file() and p.suffix in SUPPORTED_IMAGE_EXTS]
        labels_dir = sub_labels if sub_labels.is_dir() else None
    else:
        image_files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix in SUPPORTED_IMAGE_EXTS]
        labels_dir = input_dir

    if not image_files:
        raise ValueError(f"No valid image files found in {input_dir}")

    # Setup output directories
    train_img_dir = out_dir / "train" / "images"
    train_lbl_dir = out_dir / "train" / "labels"
    val_img_dir = out_dir / "validation" / "images"
    val_lbl_dir = out_dir / "validation" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled_images = list(image_files)
    rng.shuffle(shuffled_images)

    total_count = len(shuffled_images)
    train_count = int(total_count * train_pct)
    val_count = total_count - train_count

    transfer_op = shutil.move if move_files else shutil.copy2

    print(f"Total images found: {total_count}")
    print(f"Splitting: {train_count} train ({train_pct*100:.1f}%), {val_count} validation ({(1-train_pct)*100:.1f}%)")

    for idx, img_path in enumerate(shuffled_images):
        is_train = idx < train_count
        target_img_dir = train_img_dir if is_train else val_img_dir
        target_lbl_dir = train_lbl_dir if is_train else val_lbl_dir

        # Copy/move image
        transfer_op(img_path, target_img_dir / img_path.name)

        # Check for corresponding label
        if labels_dir:
            lbl_candidate = labels_dir / f"{img_path.stem}.txt"
            if lbl_candidate.exists():
                transfer_op(lbl_candidate, target_lbl_dir / lbl_candidate.name)

    print(f"Dataset split complete: Output in {out_dir}")
    return train_count, val_count
