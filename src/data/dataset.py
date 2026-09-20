"""Dataset inspection and integrity verification module."""

import os
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Any
import yaml


def get_class_distribution(labels_dir: str) -> Tuple[int, int, Dict[int, int]]:
    """Scan a label directory and return total files, empty files, and class instance counts.

    Args:
        labels_dir: Path to directory containing YOLO .txt label files.

    Returns:
        tuple: (total_files, empty_files, dict of {class_id: count})
    """
    class_counts = Counter()
    total_files = 0
    empty_files = 0

    if not os.path.exists(labels_dir):
        return 0, 0, {}

    for entry in os.scandir(labels_dir):
        if entry.is_file() and entry.name.endswith(".txt"):
            total_files += 1
            has_boxes = False
            with open(entry.path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            class_counts[cls_id] += 1
                            has_boxes = True
                        except ValueError:
                            continue
            if not has_boxes:
                empty_files += 1

    return total_files, empty_files, dict(class_counts)


def verify_dataset(data_yaml_path: str = "configs/data.yaml") -> Dict[str, Any]:
    """Verify dataset integrity, matching pairs, and class distributions.

    Args:
        data_yaml_path: Path to dataset YAML configuration file.

    Returns:
        dict containing validation statistics.
    """
    yaml_path = Path(data_yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {data_yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Determine base path for dataset
    base_path = config.get("path", "")
    if base_path:
        root_dir = (yaml_path.parent / base_path).resolve()
        if not root_dir.exists():
            # Fallback to current working directory relative
            root_dir = Path(base_path).resolve()
    else:
        root_dir = yaml_path.parent

    print("=" * 60)
    print("DATASET VERIFICATION REPORT")
    print("=" * 60)
    print(f"Config path:   {yaml_path}")
    print(f"Dataset root:  {root_dir}")

    names = config.get("names", {})
    if isinstance(names, list):
        class_map = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        class_map = {int(k): v for k, v in names.items()}
    else:
        class_map = {}

    print(f"Configured classes ({len(class_map)}): {class_map}\n")

    stats = {}
    splits = [("Train", config.get("train", "train/images")),
              ("Validation", config.get("val", "validation/images"))]

    for split_name, rel_img_dir in splits:
        img_dir = (root_dir / rel_img_dir).resolve() if not Path(rel_img_dir).is_absolute() else Path(rel_img_dir)
        # Infer labels directory by replacing /images with /labels
        label_dir_str = str(img_dir).replace(f"{os.sep}images", f"{os.sep}labels")
        label_dir = Path(label_dir_str)

        img_count = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        tot_labels, empty_labels, class_counts = get_class_distribution(str(label_dir))

        print(f"[{split_name} Split]")
        print(f"  Images directory: {img_dir} (Found: {img_count})")
        print(f"  Labels directory: {label_dir} (Found: {tot_labels})")
        print(f"  Background images (no bounding boxes): {empty_labels}")
        print("  Class Instances:")
        for cls_id in sorted(class_map.keys()):
            count = class_counts.get(cls_id, 0)
            cls_name = class_map.get(cls_id, f"Class {cls_id}")
            print(f"    - {cls_id} ({cls_name}): {count}")
        print()

        stats[split_name.lower()] = {
            "images": img_count,
            "labels": tot_labels,
            "background": empty_labels,
            "class_counts": class_counts,
        }

    print("Verification complete.")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    verify_dataset()
