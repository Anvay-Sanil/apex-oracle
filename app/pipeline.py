"""Exoplanet Inspector - self-contained demo pipeline (APEX-ORACLE front-end).

Synthetic-data backed, depends only on numpy + scipy. Mirrors the real pipeline
stages: generate -> detrend -> BLS-style period search -> phase fold ->
estimate (period, depth, duration, SNR) -> heuristic classification.

The classifier here is a transparent rule-based stand-in for the trained
multimodal Transformer; it uses the same physical features the real model learns
(transit shape, secondary eclipse, odd-even depth, SNR).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

SignalKind = Literal["planet", "eb", "starspot"]


@dataclass(frozen=True)
class LightCurveConfig:
    """Immutable configuration for a synthetic TESS-like light curve."""

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


# --------------------------------------------------------------------------- #
# Stage 1: synthetic light-curve generation
# --------------------------------------------------------------------------- #
def _transit_dip(t: np.ndarray, cfg: LightCurveConfig) -> np.ndarray:
    """Return the fractional flux dip (>=0) from a transit/eclipse signal."""
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
        dip[inside] = depth * (1.0 - x[inside] / half)  # V-shaped
        phase2 = ((t - (t0 + 0.5 * period) + 0.5 * period) % period) - 0.5 * period
        x2 = np.abs(phase2)
        sec = x2 < half
        dip[sec] += 0.35 * depth * (1.0 - x2[sec] / half)  # secondary eclipse
    return dip


def generate_lightcurve(cfg: LightCurveConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate (time_days, normalized_flux) for the configured signal."""
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.total_days * 24 * 60 / cfg.cadence_min)
    t = np.linspace(0.0, cfg.total_days, n)
    flux = np.ones(n)

    # stellar variability: quasi-periodic rotation (spots) at a few-day period
    rot = rng.uniform(4.0, 11.0)
    amp = cfg.variability_ppm * 1e-6
    flux += amp * np.sin(2 * np.pi * t / rot + rng.uniform(0, 2 * np.pi))
    flux += 0.4 * amp * np.sin(2 * np.pi * t / (rot / 2.0) + rng.uniform(0, 2 * np.pi))

    if cfg.kind in ("planet", "eb"):
        flux *= 1.0 - _transit_dip(t, cfg)

    flux += rng.normal(0.0, cfg.noise_ppm * 1e-6, n)

    if cfg.add_gap:  # mimic a perigee downlink gap
        g0 = int(0.48 * n)
        g1 = int(0.52 * n)
        keep = np.ones(n, dtype=bool)
        keep[g0:g1] = False
        t, flux = t[keep], flux[keep]

    return t, flux


# --------------------------------------------------------------------------- #
# Stage 2: detrending (transit-safe, Savitzky-Golay over a wide window)
# --------------------------------------------------------------------------- #
def detrend(
    t: np.ndarray, flux: np.ndarray, window_hours: float = 24.0, cadence_min: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten slow variability while preserving short transits. Returns (flat, trend).

    The trend is fit on a transit-masked copy (deep dips replaced by a rolling
    median) so the Savitzky-Golay filter does not absorb the transit - the same
    'do not let detrending eat the signal' guardrail used in the full pipeline.
    """
    win = int(window_hours * 60 / cadence_min)
    if win % 2 == 0:
        win += 1
    win = max(5, min(win, len(flux) - (1 - len(flux) % 2)))

    # mask points well below the local level before fitting the trend
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-12
    masked = flux.copy()
    deep = flux < med - 3.0 * 1.4826 * mad
    masked[deep] = med
    trend = savgol_filter(masked, win, polyorder=2)
    flat = flux / trend
    return flat, trend


# --------------------------------------------------------------------------- #
# Stage 3: BLS-style period search (fold-and-bin)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SearchResult:
    period: float
    t0: float
    periods: np.ndarray
    power: np.ndarray


def bls_search(
    t: np.ndarray,
    flux: np.ndarray,
    pmin: float = 0.5,
    pmax: float = 10.0,
    n_periods: int = 3000,
    nbins: int = 200,
) -> SearchResult:
    """Box-least-squares-style search: peak power = deepest persistent phase bin."""
    periods = np.linspace(pmin, pmax, n_periods)
    f = flux - np.median(flux)
    power = np.zeros(n_periods)
    for i, period in enumerate(periods):
        idx = (((t % period) / period) * nbins).astype(int) % nbins
        sums = np.bincount(idx, weights=f, minlength=nbins)
        counts = np.bincount(idx, minlength=nbins)
        means = sums / np.maximum(counts, 1)
        # weight depth by sqrt(in-bin count) so sparsely-populated bins don't win
        power[i] = (-means.min()) * np.sqrt(counts[means.argmin()])

    best_period = float(periods[power.argmax()])
    # recover epoch t0 from the deepest phase bin at the best period
    idx = (((t % best_period) / best_period) * nbins).astype(int) % nbins
    sums = np.bincount(idx, weights=f, minlength=nbins)
    counts = np.bincount(idx, minlength=nbins)
    low_bin = (sums / np.maximum(counts, 1)).argmin()
    t0 = (low_bin + 0.5) / nbins * best_period
    return SearchResult(best_period, t0, periods, power)


# --------------------------------------------------------------------------- #
# Stage 4: phase folding + parameter estimation
# --------------------------------------------------------------------------- #
def phase_fold(
    t: np.ndarray, flux: np.ndarray, period: float, t0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fold onto [-0.5, 0.5] phase, centred on transit, sorted by phase."""
    phase = ((((t - t0 + 0.5 * period) % period) - 0.5 * period)) / period
    order = np.argsort(phase)
    return phase[order], flux[order]


def _binned_profile(
    phase: np.ndarray, flux: np.ndarray, nbins: int = 120
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-0.5, 0.5, nbins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
    sums = np.bincount(idx, weights=flux, minlength=nbins)
    counts = np.bincount(idx, minlength=nbins)
    prof = sums / np.maximum(counts, 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof


@dataclass(frozen=True)
class TransitParams:
    period_days: float
    depth_ppm: float
    duration_hours: float
    snr: float
    secondary_ppm: float
    odd_even_diff_ppm: float
    flatness: float  # 1.0 ~ flat-bottom (U), low ~ pointed (V)
    oot_rms_ppm: float  # out-of-transit profile scatter (high => variable star)


def estimate_params(
    phase: np.ndarray, flux: np.ndarray, period: float
) -> TransitParams:
    """Estimate depth, duration, SNR and shape diagnostics from the folded curve."""
    centers, prof = _binned_profile(phase, flux)
    oot = np.abs(centers) > 0.15
    baseline = float(np.median(prof[oot]))
    scatter = float(np.std(flux[np.abs(phase) > 0.15]) + 1e-12)

    near = np.abs(centers) < 0.03
    depth = max(baseline - float(np.min(prof[near])), 0.0)

    # duration: phase width where the binned dip exceeds half its depth
    dipped = (baseline - prof) > 0.5 * depth
    central = dipped & (np.abs(centers) < 0.25)
    width_phase = central.sum() / len(centers)
    duration_hours = width_phase * period * 24.0

    n_in = max(1, int(np.sum(np.abs(phase) < width_phase / 2)))
    snr = depth / (scatter / np.sqrt(n_in))

    # secondary eclipse near phase 0.5 (folded edges)
    sec_mask = np.abs(np.abs(centers) - 0.5) < 0.03
    secondary = baseline - float(np.min(prof[sec_mask])) if sec_mask.any() else 0.0
    secondary = max(secondary, 0.0)

    # odd-even depth difference (alternating transits)
    odd_even = _odd_even_depth(phase, flux, baseline)

    # flatness: mean dip in inner third vs depth (U ~ 1, V ~ 0.5)
    inner = np.abs(centers) < (width_phase / 3 + 1e-3)
    flatness = float(np.mean(baseline - prof[inner]) / depth) if depth > 0 else 0.0

    return TransitParams(
        period_days=period,
        depth_ppm=depth * 1e6,
        duration_hours=duration_hours,
        snr=float(snr),
        secondary_ppm=secondary * 1e6,
        odd_even_diff_ppm=odd_even * 1e6,
        flatness=flatness,
        oot_rms_ppm=float(np.std(prof[oot]) * 1e6),
    )


def _odd_even_depth(phase: np.ndarray, flux: np.ndarray, baseline: float) -> float:
    """Crude odd-even depth difference using the in-transit points only."""
    intr = np.abs(phase) < 0.02
    if intr.sum() < 6:
        return 0.0
    depths = baseline - flux[intr]
    half = len(depths) // 2
    return abs(float(np.median(depths[:half]) - np.median(depths[half:])))


# --------------------------------------------------------------------------- #
# Stage 5: heuristic classification (stand-in for the trained model)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Classification:
    label: str
    confidence: float
    reasons: list[str]


def classify(params: TransitParams) -> Classification:
    """Rule-based transit / eclipsing binary / starspot decision with reasons."""
    reasons: list[str] = []

    localized = params.depth_ppm > 2.5 * params.oot_rms_ppm
    if params.snr < 8.0 or params.depth_ppm < 50 or not localized:
        if not localized:
            reasons.append("dip not localized - flux varies across the whole phase (rotation)")
        else:
            reasons.append(f"low SNR ({params.snr:.1f}) - no convincing transit")
        return Classification("Starspot / no clear transit", float(np.clip(0.6 + min(params.snr, 8) * 0.03, 0.6, 0.9)), reasons)

    eb_score = 0.0
    if params.secondary_ppm > 0.25 * params.depth_ppm:
        eb_score += 0.5
        reasons.append("secondary eclipse present")
    if params.flatness < 0.7:
        eb_score += 0.3
        reasons.append(f"V-shaped profile (flatness {params.flatness:.2f})")
    if params.depth_ppm > 30000:
        eb_score += 0.3
        reasons.append("very deep (>3%) eclipse")
    if params.odd_even_diff_ppm > 0.2 * params.depth_ppm:
        eb_score += 0.2
        reasons.append("odd-even depth mismatch")

    if eb_score >= 0.5:
        return Classification("Eclipsing binary / blend", min(0.55 + eb_score / 2, 0.98), reasons)

    reasons.append(f"U-shaped, achromatic dip (flatness {params.flatness:.2f}), SNR {params.snr:.1f}")
    confidence = float(np.clip(0.7 + (params.snr - 7) * 0.01 + (params.flatness - 0.7) * 0.3, 0.7, 0.985))
    return Classification("Planet transit", confidence, reasons)


def attention_weights(phase: np.ndarray, duration_phase: float) -> np.ndarray:
    """Placeholder for model attention: emphasise in-transit points (Gaussian)."""
    sigma = max(duration_phase / 2.0, 1e-3)
    return np.exp(-0.5 * (phase / sigma) ** 2)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InspectionResult:
    t: np.ndarray
    flux_raw: np.ndarray
    trend: np.ndarray
    flux_flat: np.ndarray
    search: SearchResult
    phase: np.ndarray
    flux_folded: np.ndarray
    params: TransitParams
    classification: Classification


def run_pipeline(cfg: LightCurveConfig) -> InspectionResult:
    """Full demo pipeline on a synthetic light curve."""
    t, flux_raw = generate_lightcurve(cfg)
    flux_flat, trend = detrend(t, flux_raw, cadence_min=cfg.cadence_min)
    search = bls_search(t, flux_flat)
    phase, flux_folded = phase_fold(t, flux_flat, search.period, search.t0)
    params = estimate_params(phase, flux_folded, search.period)
    classification = classify(params)
    logger.info("Recovered period=%.4f d (true=%.4f)", search.period, cfg.period_days)
    return InspectionResult(
        t, flux_raw, trend, flux_flat, search, phase, flux_folded, params, classification
    )
