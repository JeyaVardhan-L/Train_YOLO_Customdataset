#!/usr/bin/env python3
"""CLI Script to split images and YOLO labels into train and validation sets."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import split_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split image and label dataset into train/validation folders for YOLO training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to folder containing raw images and annotations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Target output directory to populate train/ and validation/ (default: 'data')",
    )
    parser.add_argument(
        "--train-pct",
        dest="train_pct",
        type=float,
        default=0.8,
        help="Fraction of dataset allocated to training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible partition (default: 42)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them (default: copy)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        train_count, val_count = split_dataset(
            input_path=args.datapath,
            output_path=args.output,
            train_pct=args.train_pct,
            seed=args.seed,
            move_files=args.move,
        )
        print(f"Successfully generated splits: {train_count} train, {val_count} validation.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
