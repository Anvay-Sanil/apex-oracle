"""End-to-end orchestration: load -> detrend -> search -> fold -> features -> classify."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .data import LightCurve
from .features import TransitFeatures, extract_features
from .model import BaseClassifier, Prediction, RuleBasedClassifier
from .preprocess import detrend, phase_fold
from .search import _fold_power, bls_search
from .utils import get_logger

logger = get_logger(__name__)


def _cadence_min(time: np.ndarray) -> float:
    dt = np.median(np.diff(time))
    return float(dt * 1440.0) if np.isfinite(dt) and dt > 0 else 10.0


def _per_point_noise(flat: np.ndarray) -> float:
    if flat.size < 3:
        return 1e-4
    return float(np.median(np.abs(np.diff(flat))) / 1.128) or 1e-4


def _extract(lc, flat, period, t0, cfg, ppn):
    phase, folded = phase_fold(lc.time, flat, period, t0)
    feats = extract_features(
        phase, folded, period, n_bins=cfg.n_phase_bins,
        effective_in_transit=cfg.effective_in_transit, per_point_noise=ppn,
    )
    return phase, folded, feats


def detect(lc: LightCurve, config: PipelineConfig | None = None):
    """Detrend -> search -> fold -> features, with equal-depth secondary de-aliasing.

    A 'secondary' as deep as the primary almost always means the search locked onto
    2x the true period (the primary transit reappears at phase 0.5). When detected,
    the period is halved and the curve re-folded.

    Returns (flat, trend, period, t0, phase, folded, features, search).
    """
    cfg = config or PipelineConfig()
    flat, trend = detrend(lc.time, lc.flux, cfg.detrend, cadence_min=_cadence_min(lc.time))
    search = bls_search(lc.time, flat, cfg.search)
    ppn = _per_point_noise(flat)
    period, t0 = search.period, search.t0
    phase, folded, feats = _extract(lc, flat, period, t0, cfg, ppn)

    if (feats.secondary_ppm > 0.6 * feats.depth_ppm
            and feats.depth_ppm > cfg.classify.depth_min_ppm):
        f = flat - np.median(flat)
        for d in (2, 3, 4):                       # try sub-multiples; adopt shortest clean one
            ph = period / d
            if ph < cfg.search.period_min:
                break
            _, mi = _fold_power(lc.time, f, ph, cfg.search.n_bins)
            t0h = (mi + 0.5) / cfg.search.n_bins * ph
            ph_phase, ph_folded, feats_h = _extract(lc, flat, ph, t0h, cfg, ppn)
            if feats_h.localized and feats_h.secondary_ppm < 0.5 * max(feats_h.depth_ppm, 1.0):
                logger.info("de-aliased period %.4f -> %.4f d (/%d, equal-depth secondary)",
                            period, ph, d)
                period, t0, phase, folded, feats = ph, t0h, ph_phase, ph_folded, feats_h
                break

    return flat, trend, period, t0, phase, folded, feats, search


def features_from_lightcurve(
    lc: LightCurve, config: PipelineConfig | None = None
) -> TransitFeatures:
    """Detrend -> search -> fold -> features. Shared by training and inference."""
    return detect(lc, config)[6]


@dataclass(frozen=True)
class InspectionResult:
    source: str
    prediction: Prediction
    features: TransitFeatures
    period_days: float
    t0_days: float
    parameters: dict
    vetting: dict
    arrays: dict = field(repr=False, default_factory=dict)

    def summary(self) -> str:
        p, f = self.prediction, self.features
        return (f"[{self.source}] {p.label.upper()} ({p.confidence:.0%}) | "
                f"P={f.period_days:.3f}±{f.period_err_days:.3f} d, "
                f"depth={f.depth_ppm:.0f}±{f.depth_err_ppm:.0f} ppm, "
                f"dur={f.duration_hours:.2f}±{f.duration_err_hours:.2f} h, SNR={f.snr:.1f}")


def _default_classifier(cfg: PipelineConfig) -> BaseClassifier:
    """Load the bundled real-data-trained hybrid model; fall back to rules if unavailable."""
    try:
        from .model import HybridClassifier, SklearnClassifier
        path = Path(__file__).parent / "models" / "default.joblib"
        if path.exists():
            return HybridClassifier(SklearnClassifier.load(path), cfg.classify)
    except Exception as exc:  # missing model / sklearn issue -> transparent fallback
        logger.warning("real-trained model unavailable, using rule-based: %s", exc)
    return RuleBasedClassifier(cfg.classify)


class ExoplanetPipeline:
    """Configurable detection + classification pipeline."""

    def __init__(self, config: PipelineConfig | None = None,
                 classifier: BaseClassifier | None = None):
        self.cfg = config or PipelineConfig()
        self.classifier = classifier or _default_classifier(self.cfg)

    def run(self, lc: LightCurve) -> InspectionResult:
        flat, _, period, t0, phase, folded, feats, search = detect(lc, self.cfg)
        pred = self.classifier.predict_one(feats)
        params = {
            "period_days": (feats.period_days, feats.period_err_days),
            "depth_ppm": (feats.depth_ppm, feats.depth_err_ppm),
            "duration_hours": (feats.duration_hours, feats.duration_err_hours),
            "snr": feats.snr,
        }
        vetting = {
            "odd_even_flag": feats.odd_even_ppm > 0.2 * max(feats.depth_ppm, 1),
            "secondary_flag": feats.secondary_ppm > 0.3 * max(feats.depth_ppm, 1),
            "localized": feats.localized,
            "centroid": "requires pixel data (not available for 1-D light curves)",
        }
        result = InspectionResult(
            source=lc.source, prediction=pred, features=feats,
            period_days=period, t0_days=t0,
            parameters=params, vetting=vetting,
            arrays={"time": lc.time, "flat": flat, "phase": phase,
                    "folded": folded, "spectrum": (search.periods, search.spectrum)},
        )
        logger.info(result.summary())
        return result
