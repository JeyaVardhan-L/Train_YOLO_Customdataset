"""YOLO Airborne Object Detection Package.

Detects aerial objects (aircraft, birds, drones) using Ultralytics YOLO models.
"""

from .inference.detector import YOLOAirDetector
from .inference.radar import RadarHUD
from .data.dataset import verify_dataset, get_class_distribution
from .data.split import split_dataset

__version__ = "1.0.0"
__all__ = [
    "YOLOAirDetector",
    "RadarHUD",
    "verify_dataset",
    "get_class_distribution",
    "split_dataset",
]
