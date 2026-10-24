"""Shared utility functions for radiology summarization."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.summarization.config import (
    LOGS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def log_message(message: str) -> None:
    """Print a consistently formatted log message."""
    print(f"[INFO] {message}")


def initialize_project() -> None:
    """Create the directories required by the summarization module."""
    create_directories(
        [
            RAW_DATA_DIR,
            PROCESSED_DATA_DIR,
            MODELS_DIR,
            OUTPUTS_DIR,
            LOGS_DIR,
        ]
    )
    log_message("Summarization directories initialized.")