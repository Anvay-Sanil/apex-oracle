"""Transit light-curve fitting: a trapezoid model fit to the folded curve.

Satisfies the requirement to estimate depth/duration *by light-curve fitting* with proper
uncertainties (from the covariance matrix), not just feature readout. scipy-only; `batman`
can refine this later but is not required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class TransitFit:
    period_days: float
    depth_ppm: float
    depth_err_ppm: float
    duration_hours: float
    duration_err_hours: float
    t0_phase: float
    reduced_chi2: float
    converged: bool


def trapezoid(phase: np.ndarray, t0: float, depth: float, half_dur: float, ingress: float) -> np.ndarray:
    """Normalised flux of a trapezoidal transit centred at phase t0 (all in phase units)."""
    x = np.abs(phase - t0)
    ingress = max(ingress, 1e-4)
    inner = max(half_dur - ingress, 0.0)
    flux = np.ones_like(phase)
    full = x < inner
    ramp = (x >= inner) & (x < half_dur)
    flux[full] = 1.0 - depth
    flux[ramp] = 1.0 - depth * (half_dur - x[ramp]) / ingress
    return flux


def fit_transit(
    phase: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    init_depth: float,
    init_dur_phase: float,
    per_point_noise: float,
) -> TransitFit:
    """Least-squares fit of a trapezoid transit to the folded light curve.

    `init_depth` is a fraction (e.g. 0.001 for 1000 ppm); `init_dur_phase` is the full transit
    width in phase. Returns fitted depth/duration with 1-sigma errors from the covariance.
    """
    half0 = max(init_dur_phase / 2.0, 1.5 / 240)
    p0 = [0.0, max(init_depth, 1e-5), half0]
    bounds = ([-0.05, 0.0, 1.0 / 240], [0.05, 0.2, 0.3])
    sigma = max(per_point_noise, 1e-6)

    def model3(ph, t0, depth, half_dur):  # ingress tied to width -> avoids depth/ingress degeneracy
        return trapezoid(ph, t0, depth, half_dur, 0.25 * half_dur)

    try:
        popt, pcov = curve_fit(
            model3, phase, flux, p0=p0, bounds=bounds, maxfev=8000,
            sigma=np.full_like(flux, sigma), absolute_sigma=True,
        )
        perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
        converged = bool(np.all(np.isfinite(perr)))
    except Exception:
        popt = np.array(p0)
        perr = np.zeros(3)
        converged = False

    t0, depth, half_dur = popt
    d_err, hd_err = perr[1], perr[2]
    if not np.isfinite(d_err) or d_err > 5 * max(depth, 1e-9):   # guard covariance blow-up
        d_err, converged = max(depth, 1e-9), False
    model = model3(phase, *popt)
    dof = max(len(flux) - 3, 1)
    red_chi2 = float(np.sum(((flux - model) / sigma) ** 2) / dof)

    return TransitFit(
        period_days=period_days,
        depth_ppm=float(depth * 1e6),
        depth_err_ppm=float(d_err * 1e6),
        duration_hours=float(2 * half_dur * period_days * 24.0),
        duration_err_hours=float(2 * hd_err * period_days * 24.0),
        t0_phase=float(t0),
        reduced_chi2=red_chi2,
        converged=bool(converged),
    )
