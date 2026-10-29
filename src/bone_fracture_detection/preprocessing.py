"""Image preprocessing shared by training and inference."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

from src.bone_fracture_detection.config import XRAY_IMAGE_SIZE


def preprocess_image(image: Image.Image, *, augment: bool = False) -> torch.Tensor:
    """Convert to a normalized grayscale tensor, with optional train augmentation."""
    width, height = XRAY_IMAGE_SIZE
    grayscale = ImageOps.pad(
        image.convert("L"),
        (width, height),
        method=Image.Resampling.BILINEAR,
        color=0,
        centering=(0.5, 0.5),
    )
    if augment:
        # Small exposure and positioning changes improve robustness without
        # mirroring laterality markers or substantially altering anatomy.
        grayscale = grayscale.rotate(
            random.uniform(-7.0, 7.0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
        grayscale = ImageEnhance.Contrast(grayscale).enhance(
            random.uniform(0.9, 1.1)
        )
        grayscale = ImageEnhance.Brightness(grayscale).enhance(
            random.uniform(0.9, 1.1)
        )
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(array).unsqueeze(0)
    return image_tensor


def preprocess_legacy_image(image: Image.Image) -> torch.Tensor:
    """Reproduce v1 stretch-resize preprocessing for legacy checkpoint evaluation."""
    width, height = XRAY_IMAGE_SIZE
    grayscale = image.convert("L").resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)
