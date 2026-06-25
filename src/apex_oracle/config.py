"""Immutable configuration objects (config-driven pipeline)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetrendConfig:
    method: str = "savgol"          # "savgol" | "wotan" (if installed)
    window_hours: float = 24.0
    sigma_clip: float = 3.0


@dataclass(frozen=True)
class SearchConfig:
    method: str = "bls"             # "bls" | "tls" (if installed)
    period_min: float = 0.5
    period_max: float = 20.0
    n_periods: int = 2500
    n_bins: int = 200
    dealias: bool = True


@dataclass(frozen=True)
class ClassifyConfig:
    snr_min: float = 8.0
    depth_min_ppm: float = 50.0
    max_duration_phase: float = 0.18      # wider => rotational, not a transit
    vshape_threshold: float = 0.50        # fill factor below this => V (eclipsing binary)
    deep_eclipse_ppm: float = 15000.0     # very deep eclipse => binary
    localized_factor: float = 2.5         # depth must exceed this * out-of-transit rms
    secondary_depth_ratio: float = 0.20   # secondary/primary depth flagging an eclipse
    secondary_snr_factor: float = 3.5     # secondary must exceed this * oot rms
    odd_even_ratio: float = 0.25          # odd-even depth mismatch => binary


@dataclass(frozen=True)
class PipelineConfig:
    n_phase_bins: int = 240
    effective_in_transit: int = 64      # for SNR baseline (fixed observation depth)
    detrend: DetrendConfig = field(default_factory=DetrendConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    seed: int = 42
