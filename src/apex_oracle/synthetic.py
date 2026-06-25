"""Synthetic TESS-like light-curve generation and transit injection.

Used for the demo, for building training sets, and for injection-recovery tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SignalKind = Literal["planet", "eb", "starspot"]


@dataclass(frozen=True)
class SyntheticConfig:
    period_days: float = 3.41
    t0_days: float = 1.10
    depth_ppm: float = 900.0
    duration_hours: float = 2.7
    noise_ppm: float = 250.0
    variability_ppm: float = 1500.0
    total_days: float = 27.4
    cadence_min: float = 10.0
    kind: SignalKind = "planet"
    add_gap: bool = True
    seed: int = 42


def _transit_dip(t: np.ndarray, cfg: SyntheticConfig) -> np.ndarray:
    period, t0 = cfg.period_days, cfg.t0_days
    duration = cfg.duration_hours / 24.0
    depth = cfg.depth_ppm * 1e-6
    half = duration / 2.0
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    x = np.abs(phase)
    dip = np.zeros_like(t)
    if cfg.kind == "planet":
        ingress = 0.2 * duration
        flat = x < (half - ingress)
        ramp = (x >= (half - ingress)) & (x < half)
        dip[flat] = depth
        dip[ramp] = depth * (half - x[ramp]) / ingress
    elif cfg.kind == "eb":
        inside = x < half
        dip[inside] = depth * (1.0 - x[inside] / half)              # V-shaped
        phase2 = ((t - (t0 + 0.5 * period) + 0.5 * period) % period) - 0.5 * period
        x2 = np.abs(phase2)
        sec = x2 < half
        dip[sec] += 0.35 * depth * (1.0 - x2[sec] / half)           # secondary eclipse
    return dip


def generate_lightcurve(cfg: SyntheticConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_days, normalized_flux) for the configured signal."""
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.total_days * 24 * 60 / cfg.cadence_min)
    t = np.linspace(0.0, cfg.total_days, n)
    flux = np.ones(n)
    rot = rng.uniform(4.0, 11.0)
    amp = cfg.variability_ppm * 1e-6
    flux += amp * np.sin(2 * np.pi * t / rot + rng.uniform(0, 2 * np.pi))
    flux += 0.4 * amp * np.sin(2 * np.pi * t / (rot / 2.0) + rng.uniform(0, 2 * np.pi))
    if cfg.kind in ("planet", "eb"):
        flux *= 1.0 - _transit_dip(t, cfg)
    flux += rng.normal(0.0, cfg.noise_ppm * 1e-6, n)
    if cfg.add_gap:
        keep = np.ones(n, dtype=bool)
        keep[int(0.48 * n):int(0.52 * n)] = False
        t, flux = t[keep], flux[keep]
    return t, flux


def inject_transit(
    t: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    t0_days: float,
    depth_ppm: float,
    duration_hours: float,
) -> np.ndarray:
    """Inject a box/trapezoid transit into an existing flux series (for inj-recovery)."""
    cfg = SyntheticConfig(
        period_days=period_days, t0_days=t0_days, depth_ppm=depth_ppm,
        duration_hours=duration_hours, kind="planet",
    )
    return flux * (1.0 - _transit_dip(t, cfg))
