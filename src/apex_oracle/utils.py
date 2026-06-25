"""Shared utilities: seeding and logging."""
from __future__ import annotations

import logging
import os
import random

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def set_seed(seed: int = 42) -> None:
    """Set process-wide random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def median_abs_dev(x: np.ndarray) -> float:
    """Robust scatter estimate (1.4826 * MAD)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))
