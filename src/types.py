"""Shared data types for the iris recognition pipeline."""

from dataclasses import dataclass
import numpy as np


@dataclass
class IrisSample:
    subject_id: str   # stable per-eye class label (e.g. "002")
    session: int      # 1 = training, 2 = test
    image_path: str   # absolute path to the .bmp file
    image: np.ndarray # grayscale uint8, shape (H, W)
