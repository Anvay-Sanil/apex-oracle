"""Transit-safe detrending and phase folding."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .config import DetrendConfig
from .utils import median_abs_dev


def detrend(
    t: np.ndarray, flux: np.ndarray, cfg: DetrendConfig, cadence_min: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten slow variability while preserving short transits.

    The trend is fit on a transit-masked copy so the filter does not absorb the
    transit. Falls back gracefully if `wotan` is unavailable.
    Returns (flattened_flux, trend).
    """
    if cfg.method == "wotan":
        try:  # pragma: no cover - optional dependency
            from wotan import flatten

            flat, trend = flatten(
                t, flux, method="biweight",
                window_length=cfg.window_hours / 24.0, return_trend=True,
            )
            return np.asarray(flat), np.asarray(trend)
        except Exception:
            pass  # fall through to savgol

    win = int(cfg.window_hours * 60 / cadence_min)
    if win % 2 == 0:
        win += 1
    win = max(5, min(win, len(flux) - (1 - len(flux) % 2)))
    med = np.median(flux)
    mad = median_abs_dev(flux) + 1e-12
    masked = flux.copy()
    masked[flux < med - cfg.sigma_clip * mad] = med
    trend = savgol_filter(masked, win, polyorder=2)
    return flux / trend, trend


def phase_fold(
    t: np.ndarray, flux: np.ndarray, period: float, t0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fold onto [-0.5, 0.5] phase, transit centred, sorted by phase."""
    phase = (((t - t0 + 0.5 * period) % period) - 0.5 * period) / period
    order = np.argsort(phase)
    return phase[order], flux[order]


def binned_profile(
    phase: np.ndarray, flux: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a folded curve. Returns (bin_centers, profile) with NaN for empty bins."""
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=flux, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    prof = np.full(n_bins, np.nan)
    nz = counts > 0
    prof[nz] = sums[nz] / counts[nz]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof
