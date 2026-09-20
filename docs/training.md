# Model Training Guide

This guide covers training YOLO object detection models on the custom airborne dataset.

## Environment Preparation

### Python & Virtual Environment

Create an isolated virtual environment (Python 3.10 to 3.12 recommended):

```bash
# Using standard venv
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate
```

### PyTorch GPU Acceleration (CUDA)

For NVIDIA GPU acceleration, install PyTorch with CUDA support:

```bash
# Example for CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU visibility:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Install repository dependencies:
```bash
pip install -r requirements.txt
```

---

## Starting Training

### 1. Using the Python CLI Script (Recommended)

```bash
python scripts/train.py \
    --data configs/data.yaml \
    --model yolo11s.pt \
    --epochs 60 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

### 2. Using the Ultralytics CLI Directly

```bash
yolo detect train data=configs/data.yaml model=yolo11s.pt epochs=60 batch=16 imgsz=640 device=0
```

---

## Training Options Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data` | `configs/data.yaml` | Path to dataset YAML configuration |
| `--model` | `yolo11s.pt` | Initial weights (`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`) |
| `--epochs` | `60` | Total number of training epochs |
| `--batch` | `16` | Batch size (reduce to 8 or 4 if encountering CUDA OOM) |
| `--imgsz` | `640` | Input image size in pixels |
| `--device` | `None` (auto) | Device ID (`0`, `0,1`, or `cpu`) |
| `--workers` | `8` | Data worker threads (set to `2` or `4` on Windows if thread issues arise) |
| `--project` | `runs/detect` | Root output directory |
| `--name` | `train` | Subdirectory name for experiment artifacts |
| `--resume` | `False` | Flag to resume interrupted training from `last.pt` |

---

## Output Artifacts

Each training run produces:
- `runs/detect/<name>/weights/best.pt`: Highest validation mAP checkpoint.
- `runs/detect/<name>/weights/last.pt`: Checkpoint from final epoch.
- `runs/detect/<name>/results.csv`: Epoch-by-epoch loss and metric history.
- `runs/detect/<name>/results.png`: Training curve plots (losses, mAP, precision, recall).
- `runs/detect/<name>/confusion_matrix.png`: Class-level classification confusion matrix.
- `runs/detect/<name>/F1_curve.png`: F1 confidence threshold curve.
