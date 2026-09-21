# YOLO11 Airborne Object Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

A computer vision project using Ultralytics YOLO11 to detect, classify, and track aerial objects (aircraft, birds, and drones). The repository provides dataset configuration, model training and validation routines, standard inference on images, video, and webcams, a live 2D tactical radar display interface, and an optional microcontroller-based ultrasonic hardware trigger.

---

## Problem Statement

Low-altitude airspace monitoring and perimeter surveillance face two persistent challenges:
1. Distinguishing small commercial drones from avian wildlife (birds) at distance.
2. Detecting aerial targets against diverse and high-contrast sky backgrounds (clear, overcast, backlit).

This detector is trained specifically on aerial imagery to categorize targets into three distinct classes:
- **Aircrafts**: Fixed-wing aircraft, commercial airliners, jets, and helicopters.
- **Bird**: Flying birds across varying flight profiles (reduces false-positive drone alerts).
- **Drone**: Multi-rotor UAVs, quadcopters, and commercial drones.

---

## Dataset

The dataset consists of 8,888 paired images and normalized YOLO bounding-box annotation files:

| Class ID | Class Name | Train Instances | Validation Instances | Total Annotations | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `0` | `Aircrafts` | 4,025 | 977 | 5,002 | Airplanes, passenger jets, and helicopters |
| `1` | `Bird` | 3,826 | 1,159 | 4,985 | Avian wildlife in flight |
| `2` | `Drone` | 4,073 | 958 | 5,031 | Quadcopters, hexacopters, and commercial UAVs |
| **Total** | | **11,924** | **3,094** | **15,018** | **8,888 total images** |

- **Partition**: 7,110 training images (80.0%), 1,778 validation images (20.0%).
- **Annotation format**: Standard normalized YOLO coordinates (`<class_id> <x_center> <y_center> <width> <height>`).
- **Source**: Roboflow Universe (`drone-detection-axkso` v3 by `subhranil-dey-wttsv`), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Detailed specifications are documented in [docs/dataset.md](docs/dataset.md).

---

## Pipeline Architecture

```
  Raw Aerial Dataset (8,888 Samples)
      │
      ▼
  [Dataset Preparation]  ──► scripts/train_val_split.py (80% Train / 20% Val)
      │
      ▼
  [YOLO Configuration]   ──► configs/data.yaml (Classes: Aircrafts, Bird, Drone)
      │
      ▼
  [Model Training]       ──► scripts/train.py (YOLO11s, 60 epochs, CUDA)
      │
      ▼
  [Validation & Eval]    ──► scripts/val.py (mAP@50: 0.900, mAP@50-95: 0.565)
      │
      ├──► Standard Inference ──► scripts/detect.py (Image / Folder / Video / Webcam)
      │
      └──► Tactical Radar HUD ──► scripts/radar_detect.py & scripts/hardware_trigger.py
```

---

## Verified Benchmarks

Evaluation performed on the 1,778-sample validation split at $640 \times 640$ resolution:

| Run Name | Architecture | Epochs | Precision | Recall | mAP@50 | mAP@50-95 | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Run 1 (`train`) | YOLO11s | 35 | 0.873 | 0.833 | 0.888 | 0.550 | `runs/detect/train/weights/best.pt` |
| Run 2 (`train2`) | YOLO11s | 57 | **0.888** | **0.844** | **0.900** | **0.565** | `weights/best.pt` |

The checkpoint at `weights/best.pt` corresponds to Run 2. Full metrics, loss curves, and confusion matrix data are documented in [docs/experiments.md](docs/experiments.md).

---

## Repository Structure

```
Train_YOLO_Customdataset/
├── configs/
│   ├── data.yaml              # Portable dataset configuration
│   └── train_config.yaml      # Recommended training hyperparameters
├── src/
│   ├── data/                  # Verification and splitting utilities
│   ├── inference/             # Detection engine and Radar HUD renderer
│   └── utils/                 # Visualization palettes and drawing helpers
├── scripts/
│   ├── train.py               # CLI script to train models
│   ├── val.py                 # CLI script to evaluate checkpoints
│   ├── detect.py              # CLI script for image, folder, video, and webcam inference
│   ├── radar_detect.py        # Optical + tactical 2D radar HUD interface
│   ├── train_val_split.py     # Reusable dataset partitioning utility
│   └── hardware_trigger.py    # Arduino serial proximity trigger listener
├── tests/
│   ├── test_config.py         # Configuration and path validation tests
│   ├── test_radar.py          # Radar HUD coordinate transformation tests
│   ├── test_split.py          # Dataset split ratio tests
│   └── test_visualization.py  # Color palette and visualization tests
├── docs/
│   ├── dataset.md             # Dataset annotations and class distribution
│   ├── training.md            # Environment setup and GPU training instructions
│   ├── experiments.md         # Benchmark comparison and training logs
│   └── hardware_integration.md# Radar HUD math and Arduino hardware setup
├── assets/
│   ├── testvidairtrack.mp4    # Sample tracking video for inference demos
│   └── diagrams/              # Folder hierarchy and architectural diagrams
├── weights/
│   └── best.pt                # Trained YOLO11s weights (Run 2, mAP@50: 0.900)
├── data/                      # 8,888 paired images and label files
│   ├── train/                 # 7,110 samples
│   └── validation/            # 1,778 samples
└── runs/                      # Training run artifacts, curves, and logs
```

---

## Setup & Installation

### 1. Environment Setup

```bash
git clone https://github.com/JeyaVardhan-L/Train_YOLO_Customdataset.git
cd Train_YOLO_Customdataset

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# PyTorch with CUDA 12.4 support (recommended for NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Package dependencies
pip install -r requirements.txt
```

### 3. Verify Dataset

```bash
python -c "from src.data.dataset import verify_dataset; verify_dataset('configs/data.yaml')"
```

---

## Usage

### Running Inference

Run detection on the included sample video:
```bash
python scripts/detect.py --model weights/best.pt --source assets/testvidairtrack.mp4 --save
```

Run on an image or directory of images:
```bash
# Single image
python scripts/detect.py --model weights/best.pt --source path/to/image.jpg --conf 0.5 --save

# Directory of images
python scripts/detect.py --model weights/best.pt --source data/validation/images/ --conf 0.5 --save
```

Run on a live webcam:
```bash
python scripts/detect.py --model weights/best.pt --source 0 --conf 0.45
```

### Live Tactical Radar HUD

Launch the split-screen optical camera + polar radar scope interface:
```bash
# On webcam (device index 0)
python scripts/radar_detect.py --model weights/best.pt --source 0 --conf 0.45

# On video file
python scripts/radar_detect.py --model weights/best.pt --source assets/testvidairtrack.mp4
```

### Model Training

Train a YOLO11 model using the custom dataset configuration:
```bash
python scripts/train.py \
    --data configs/data.yaml \
    --model yolo11s.pt \
    --epochs 60 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

### Model Evaluation

Evaluate checkpoints against the validation split:
```bash
python scripts/val.py --model weights/best.pt --data configs/data.yaml --imgsz 640
```

### Hardware Trigger Integration (Optional)

Listen for proximity triggers from an Arduino ultrasonic sensor over serial:
```bash
python scripts/hardware_trigger.py --port COM3 --baud 9600 --trigger-cmd scripts/radar_detect.py
```

See [docs/hardware_integration.md](docs/hardware_integration.md) for circuit schematics and the Arduino sketch.

---

## Running Tests

Run the test suite:
```bash
python -m unittest discover -s tests -v
```

---

## Current Limitations

- **Aggregated Aircraft Class**: Helicopters, commercial airliners, and light aircraft are grouped under a single `Aircrafts` class due to source dataset annotations.
- **Distant Small Targets**: Targets smaller than $16 \times 16$ pixels against overcast or cloudy skies exhibit lower recall.
- **2D Radar Projection**: The radar HUD projects normalized 2D image coordinates onto a polar plane; it does not estimate true 3D elevation or slant range without depth sensor telemetry.

---

## Next Steps

- Integrate BoT-SORT / ByteTrack multi-object tracking for persistent track IDs and trajectory estimation.
- Export weights to ONNX and TensorRT for edge deployment (NVIDIA Jetson / Raspberry Pi).
- Sub-annotate `Aircrafts` to distinguish fixed-wing aircraft from helicopters.
- Add rangefinder or stereo-camera integration for true 3D spatial positioning.

---

## License & Attribution

- **Source Code**: [MIT License](LICENSE).
- **Dataset**: Sourced from Roboflow Universe (`drone-detection-axkso` v3 by `subhranil-dey-wttsv`), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **YOLO Framework**: Built with [Ultralytics](https://github.com/ultralytics/ultralytics).
