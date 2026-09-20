# Dataset Documentation: Aerial & Airborne Objects

## Overview

This dataset is designed for aerial threat and sky object detection, specifically targeting unmanned aerial vehicles (drones), aircraft (commercial airplanes and helicopters), and birds.

```
data/
├── train/
│   ├── images/   # 7,110 images (.jpg)
│   └── labels/   # 7,110 annotation files (.txt)
└── validation/
    ├── images/   # 1,778 images (.jpg)
    └── labels/   # 1,778 annotation files (.txt)
```

Total samples: **8,888 paired images and annotations** (80.0% train / 20.0% validation).

---

## Classes and Annotations

The dataset is partitioned into three distinct classes:

| Class ID | Class Name | Train Instances | Validation Instances | Total Instances | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `0` | `Aircrafts` | 4,025 | 977 | 5,002 | Fixed-wing airplanes, commercial airliners, jets, and helicopters. |
| `1` | `Bird` | 3,826 | 1,159 | 4,985 | Avian wildlife flying in sky/clouds (common false positive source for drones). |
| `2` | `Drone` | 4,073 | 958 | 5,031 | Quadcopters, multirotors, hexacopters, and commercial UAVs. |
| **Total** | | **11,924** | **3,094** | **15,018** | Balanced class distribution across train and validation splits. |

### Annotation Format

Annotations adhere strictly to the standard **Ultralytics YOLO normalized format**:

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: Integer index (`0`, `1`, or `2`).
- `x_center`, `y_center`: Bounding box center coordinates normalized to `[0.0, 1.0]` relative to image width and height.
- `width`, `height`: Bounding box dimensions normalized to `[0.0, 1.0]`.

Example line:
```
2 0.485123 0.320491 0.114200 0.089450
```

---

## Source and Attribution

- **Dataset Source**: Roboflow Universe
- **Project**: `drone-detection-axkso` (Version 3)
- **Author/Workspace**: `subhranil-dey-wttsv`
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **URL**: [https://universe.roboflow.com/subhranil-dey-wttsv/drone-detection-axkso/dataset/3](https://universe.roboflow.com/subhranil-dey-wttsv/drone-detection-axkso/dataset/3)

---

## Integrity Verification

To programmatically verify image/label pairing, check for missing files, and count class distributions:

```bash
python -c "from src.data.dataset import verify_dataset; verify_dataset('configs/data.yaml')"
```

---

## Splitting Custom Data

If adding new raw unpartitioned images and labels, use the automated splitting tool:

```bash
python scripts/train_val_split.py --datapath path/to/raw_data --output data --train-pct 0.8 --seed 42
```
