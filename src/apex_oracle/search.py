"""Period search: BLS-style fold-and-bin with de-aliasing (TLS optional)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SearchConfig


@dataclass(frozen=True)
class SearchResult:
    period: float
    t0: float
    power: float
    periods: np.ndarray
    spectrum: np.ndarray


def _fold_power(t: np.ndarray, f: np.ndarray, period: float, n_bins: int) -> tuple[float, int]:
    idx = (((t % period) / period) * n_bins).astype(int) % n_bins
    sums = np.bincount(idx, weights=f, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)
    mi = int(np.argmin(means))
    return float(-means[mi] * np.sqrt(max(counts[mi], 1))), mi


def bls_search(t: np.ndarray, flux: np.ndarray, cfg: SearchConfig) -> SearchResult:
    """Box-least-squares-style search; peak power = deepest persistent phase bin."""
    if cfg.method == "tls":
        try:  # pragma: no cover - optional dependency
            from transitleastsquares import transitleastsquares

            model = transitleastsquares(t, flux)
            res = model.power(period_min=cfg.period_min, period_max=cfg.period_max)
            return SearchResult(
                float(res.period), float(res.T0), float(res.SDE),
                np.asarray(res.periods), np.asarray(res.power),
            )
        except Exception:
            pass

    span = float(t[-1] - t[0])
    pmax = min(cfg.period_max, max(span / 2.0, cfg.period_min * 2))
    periods = np.linspace(cfg.period_min, pmax, cfg.n_periods)
    f = flux - np.median(flux)
    spectrum = np.array([_fold_power(t, f, p, cfg.n_bins)[0] for p in periods])
    best_i = int(np.argmax(spectrum))
    period = float(periods[best_i])
    best_pw = float(spectrum[best_i])

    if cfg.dealias:  # prefer the shortest period that keeps >=85% of the power
        for div in (3, 2):
            pd = period / div
            if pd >= cfg.period_min:
                pw, _ = _fold_power(t, f, pd, cfg.n_bins)
                if pw >= 0.85 * best_pw:
                    period = pd
                    break

    _, mi = _fold_power(t, f, period, cfg.n_bins)
    t0 = (mi + 0.5) / cfg.n_bins * period
    return SearchResult(period, t0, best_pw, periods, spectrum)
