# YOLO11 Airborne Object Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

A complete, production-ready computer vision pipeline using **Ultralytics YOLO11** for detecting, classifying, and tracking aerial targets (aircraft, birds, and drones). Includes custom dataset configuration, training workflows, model evaluation, standard detection, live 2D Radar HUD visualization, and hardware trigger integration.

---

## Architecture and Pipeline Flow

```
+--------------------------------------------------------------------------------+
|                               ML Pipeline Flow                                 |
+--------------------------------------------------------------------------------+
  Raw Data (Images & Labels)
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
  [Validation & Eval]    ──► scripts/val.py (mAP@50: 90.0%, mAP@50-95: 56.5%)
      │
      ├──► Standard Inference ──► scripts/detect.py (Image / Folder / Video / Webcam)
      │
      └──► Tactical Radar HUD ──► scripts/radar_detect.py (Polar scope target tracker)
```

---

## Key Features

- **Trained YOLO11 Model**: Custom checkpoint trained for 57 epochs, achieving **90.0% mAP@50** and **88.8% precision**.
- **Balanced Aerial Dataset**: 8,888 paired images and normalized YOLO bounding-box annotations across 3 categories.
- **Modular Codebase**: Decoupled `src/` core package with clean CLI entry points under `scripts/`.
- **Bug Fixes & Portability**: Fully portable paths (eliminates machine-specific hardcoded paths) and corrected confidence thresholding logic.
- **Tactical Radar HUD**: Real-time OpenCV visualization overlaying bounding-box detections onto a 2D polar radar scope.
- **Perimeter Hardware Trigger**: Arduino ultrasonic sensor integration script for automated trigger-on-detect surveillance.

---

## Target Classes & Dataset Statistics

The dataset targets low-altitude and airspace threats, addressing the critical challenge of distinguishing small drones from birds and distant aircraft:

| Class ID | Class Name | Train Instances | Validation Instances | Total Annotations | Scope |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `0` | `Aircrafts` | 4,025 | 977 | 5,002 | Airplanes, passenger jets, and helicopters |
| `1` | `Bird` | 3,826 | 1,159 | 4,985 | Avian wildlife (reduces drone false positives) |
| `2` | `Drone` | 4,073 | 958 | 5,031 | Quadcopters, multirotors, and commercial UAVs |
| **Total** | | **11,924** | **3,094** | **15,018** | **8,888 total images** |

Dataset origin: Roboflow Universe (`drone-detection-axkso` v3 by `subhranil-dey-wttsv`), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [docs/dataset.md](docs/dataset.md) for full dataset specifications.

---

## Performance Benchmarks

Evaluated on the 1,778-sample validation split using image resolution $640 \times 640$:

| Model Checkpoint | Architecture | Epochs | Precision | Recall | mAP@50 | mAP@50-95 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `runs/detect/train/weights/best.pt` | YOLO11s | 35 | 87.3% | 83.3% | 88.8% | 55.0% | Run 1 Baseline |
| `weights/best.pt` (`train2`) | YOLO11s | 57 | **88.8%** | **84.4%** | **90.0%** | **56.5%** | **Active Best Checkpoint** |

See [docs/experiments.md](docs/experiments.md) for detailed loss curves, F1 confidence curves, and confusion matrix analysis.

---

## Directory Structure

```
Train_YOLO_Customdataset/
├── configs/
│   ├── data.yaml                      # Portable dataset configuration
│   └── train_config.yaml              # Recommended training hyperparameters
├── src/
│   ├── data/                          # Dataset validation and split utilities
│   ├── inference/                     # Detector engine and Radar HUD renderer
│   └── utils/                         # Color palettes and visualization helpers
├── scripts/
│   ├── train.py                       # CLI script to train models
│   ├── val.py                         # CLI script to evaluate checkpoints
│   ├── detect.py                      # CLI script for image/video/webcam inference
│   ├── radar_detect.py                # Live optical + tactical radar HUD interface
│   ├── train_val_split.py             # Reusable dataset partitioning utility
│   └── hardware_trigger.py            # Arduino serial trigger listener
├── docs/
│   ├── dataset.md                     # Class distribution and annotation format
│   ├── training.md                    # Environment and GPU training instructions
│   ├── experiments.md                 # Run 1 vs Run 2 benchmark comparison
│   └── hardware_integration.md        # Radar HUD and ultrasonic sensor guide
├── assets/
│   ├── testvidairtrack.mp4            # Sample flight tracking video
│   └── diagrams/                      # Workflow and diagram assets
├── weights/
│   └── best.pt                        # Custom trained model (mAP@50: 90.0%)
├── data/                              # Preserved dataset (8,888 samples)
│   ├── train/                         # 7,110 images and labels
│   └── validation/                    # 1,778 images and labels
└── runs/                              # Experimental logs, plots, and checkpoints
```

---

## Installation & Setup

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/JeyaVardhan-L/Train_YOLO_Customdataset.git
cd Train_YOLO_Customdataset

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install PyTorch & Dependencies

For GPU acceleration (NVIDIA CUDA 12.4):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Install package requirements:
```bash
pip install -r requirements.txt
```

### 3. Verify Dataset Integrity

```bash
python -c "from src.data.dataset import verify_dataset; verify_dataset('configs/data.yaml')"
```

---

## Quickstart & Usage

### 1. Running Inference

Detect airborne objects in a video file:
```bash
python scripts/detect.py --model weights/best.pt --source assets/testvidairtrack.mp4 --save
```

Run on an image or directory of images:
```bash
# Single image
python scripts/detect.py --model weights/best.pt --source path/to/image.jpg --conf 0.5 --save

# Image directory
python scripts/detect.py --model weights/best.pt --source data/validation/images/ --conf 0.5 --save
```

Run on a connected USB webcam:
```bash
python scripts/detect.py --model weights/best.pt --source 0 --conf 0.4
```

### 2. Live Tactical Radar HUD

Launch the split-screen optical camera + polar radar scope:
```bash
# On webcam (device 0)
python scripts/radar_detect.py --model weights/best.pt --source 0 --conf 0.45

# On video file
python scripts/radar_detect.py --model weights/best.pt --source assets/testvidairtrack.mp4
```

### 3. Model Training

Train a YOLO11 model using the dataset configuration:
```bash
python scripts/train.py \
    --data configs/data.yaml \
    --model yolo11s.pt \
    --epochs 60 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

### 4. Model Evaluation

Evaluate checkpoints on the validation split:
```bash
python scripts/val.py --model weights/best.pt --data configs/data.yaml --imgsz 640
```

### 5. Hardware Trigger (Arduino + Ultrasonic)

Listen for hardware proximity events over serial to trigger recording:
```bash
python scripts/hardware_trigger.py --port COM3 --baud 9600 --trigger-cmd scripts/radar_detect.py
```

---

## Current Limitations & Roadmap

### Current Limitations
- **Fine-Grained Classification**: Helicopters and fixed-wing airplanes are currently aggregated into a single `Aircrafts` class.
- **Long-Distance Detection**: Objects occupying less than $16 \times 16$ pixels at extreme distances or against overcast sky backgrounds exhibit lower recall.
- **2D Scope Projection**: The radar HUD projects normalized 2D image coordinates and does not infer real 3D elevation without stereo vision or rangefinder telemetry.

### Roadmap
- [ ] Implement BoT-SORT / ByteTrack multi-object tracking for persistent target IDs across occlusion.
- [ ] Export trained checkpoints to ONNX and TensorRT for low-latency edge deployment (NVIDIA Jetson / Raspberry Pi).
- [ ] Expand dataset annotations to split `Aircrafts` into `Airplane` and `Helicopter`.
- [ ] Integrate depth camera (Intel RealSense) for true 3D spatial radar mapping.

---

## License & Attribution

- **Source Code**: Released under the [MIT License](LICENSE).
- **Dataset**: Created by `subhranil-dey-wttsv` via Roboflow, distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **YOLO Implementation**: Powered by [Ultralytics](https://github.com/ultralytics/ultralytics).
