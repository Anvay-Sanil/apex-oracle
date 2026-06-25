"""Physical feature extraction from a phase-folded light curve.

Mirrors the verified browser-demo logic with robust, median-based estimators.
These features feed both the rule-based and the trained classifiers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .preprocess import binned_profile


@dataclass(frozen=True)
class TransitFeatures:
    period_days: float
    depth_ppm: float
    duration_hours: float
    snr: float
    sharpness: float        # deep-width / half-width: U ~ 1, V ~ < 0.4
    secondary_ppm: float
    odd_even_ppm: float
    oot_rms_ppm: float
    duration_phase: float
    localized: bool
    # 1-sigma uncertainties
    depth_err_ppm: float
    period_err_days: float
    duration_err_hours: float

    def to_vector(self) -> np.ndarray:
        """Feature vector for ML classifiers."""
        d = self.depth_ppm if self.depth_ppm > 0 else 1.0
        return np.array([
            self.depth_ppm, self.duration_hours, self.snr, self.sharpness,
            self.secondary_ppm / d, self.odd_even_ppm / d,
            self.oot_rms_ppm, self.duration_phase,
        ], dtype=float)

    @staticmethod
    def vector_names() -> list[str]:
        return ["depth_ppm", "duration_h", "snr", "sharpness",
                "secondary_ratio", "odd_even_ratio", "oot_rms_ppm", "dur_phase"]

    def as_dict(self) -> dict:
        return asdict(self)


def _smooth(prof: np.ndarray, k: int = 2) -> np.ndarray:
    """NaN-aware moving average (window 2k+1) to stabilise sparse-bin estimates."""
    out = prof.copy()
    n = prof.size
    for i in range(n):
        seg = prof[max(0, i - k):min(n, i + k + 1)]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def extract_features(
    phase: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    n_bins: int = 240,
    effective_in_transit: int = 64,
    per_point_noise: float | None = None,
) -> TransitFeatures:
    """Compute physical + shape features from a folded, transit-centred light curve.

    Robust estimators: a smoothed profile, an area-based 'fill factor' for U-vs-V
    shape, and a minimum-based secondary-eclipse depth.
    """
    centers, raw = binned_profile(phase, flux, n_bins)
    prof = _smooth(raw, k=2)
    c = np.abs(centers)
    valid = ~np.isnan(prof)

    oot = prof[(c > 0.15) & valid]
    baseline = float(np.median(oot)) if oot.size else 1.0
    oot_rms = float(np.std(oot)) if oot.size > 1 else 1e-6
    d_arr = baseline - prof

    center = (c < 0.06) & valid
    depth0 = max(baseline - float(np.min(prof[center])) if center.any() else 0.0, 1e-9)

    # full transit window (down to 20% depth captures V wings), then robust floor
    full = (c < 0.22) & valid & (d_arr > 0.2 * depth0)
    dur_phase = max(int(full.sum()) / n_bins, 2.0 / n_bins)
    half = dur_phase / 2.0
    core = (c < max(0.35 * half, 1.0 / n_bins)) & valid
    depth = max(baseline - (float(np.median(prof[core])) if core.any() else baseline), 0.0)

    # sharpness = area fill factor over the transit window: box/U ~ 0.8, V ~ 0.55
    fill_d = d_arr[full]
    sharpness = float(np.clip(np.mean(fill_d) / depth, 0.0, 1.2)) if (depth > 0 and full.any()) else 0.0

    sec_region = (c >= 0.42) & valid
    secondary = max(baseline - (float(np.min(prof[sec_region])) if sec_region.any() else baseline), 0.0)

    intr = (np.abs(phase) < 0.02)
    odd_even = 0.0
    if intr.sum() >= 6:
        dep = baseline - flux[intr]
        h = dep.size // 2
        odd_even = abs(float(np.median(dep[:h]) - np.median(dep[h:])))

    pp = per_point_noise if per_point_noise is not None else oot_rms
    pp = pp or 1e-6
    snr = depth / (pp / np.sqrt(effective_in_transit))
    localized = depth > 2.5 * oot_rms

    depth_err = pp * 1e6 / np.sqrt(effective_in_transit)
    dur_err = (1.0 / n_bins) * period_days * 24.0
    period_err = max(0.001, period_days / n_bins)

    return TransitFeatures(
        period_days=period_days,
        depth_ppm=depth * 1e6,
        duration_hours=dur_phase * period_days * 24.0,
        snr=float(snr),
        sharpness=float(sharpness),
        secondary_ppm=secondary * 1e6,
        odd_even_ppm=odd_even * 1e6,
        oot_rms_ppm=oot_rms * 1e6,
        duration_phase=float(dur_phase),
        localized=bool(localized),
        depth_err_ppm=float(depth_err),
        period_err_days=float(period_err),
        duration_err_hours=float(dur_err),
    )
