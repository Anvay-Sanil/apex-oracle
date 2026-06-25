"""Classifiers: a transparent rule-based baseline and a trainable scikit-learn model.

Both consume `TransitFeatures` and emit a `Prediction`. A factory/registry lets the
pipeline swap models by name (config-driven), per the project architecture rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import ClassifyConfig
from .features import TransitFeatures

LABELS = ("transit", "eclipse", "starspot")

_REGISTRY: dict[str, Callable[..., "BaseClassifier"]] = {}


def register_classifier(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def make_classifier(name: str, **kwargs) -> "BaseClassifier":
    if name not in _REGISTRY:
        raise KeyError(f"unknown classifier '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    reasons: list[str]
    method: str


class BaseClassifier:
    def predict_one(self, f: TransitFeatures) -> Prediction:  # pragma: no cover
        raise NotImplementedError


@register_classifier("rule")
class RuleBasedClassifier(BaseClassifier):
    """Physics-motivated rules; fully transparent (the demo's logic)."""

    def __init__(self, config: ClassifyConfig | None = None):
        self.cfg = config or ClassifyConfig()

    def predict_one(self, f: TransitFeatures) -> Prediction:
        c = self.cfg
        if (f.snr < c.snr_min or f.depth_ppm < c.depth_min_ppm
                or not f.localized or f.duration_phase > c.max_duration_phase):
            reasons = ([f"dip not localized / too broad ({f.duration_phase*100:.0f}% of phase)",
                        "consistent with rotational modulation", f"SNR {f.snr:.1f}"]
                       if (not f.localized or f.duration_phase > c.max_duration_phase)
                       else [f"low SNR ({f.snr:.1f})", "no convincing periodic dip"])
            conf = float(np.clip(0.6 + min(f.snr, c.snr_min) * 0.03, 0.6, 0.9))
            return Prediction("starspot", conf, reasons, "rule")

        eb, reasons = 0.0, []
        if (f.secondary_ppm > c.secondary_depth_ratio * f.depth_ppm
                and f.secondary_ppm > c.secondary_snr_factor * f.oot_rms_ppm):
            eb += 0.6
            reasons.append(f"secondary eclipse detected ({f.secondary_ppm:.0f} ppm)")
        if f.depth_ppm > c.deep_eclipse_ppm:
            eb += 0.5
            reasons.append("very deep (>1.5%) eclipse")
        if f.sharpness < c.vshape_threshold and f.depth_ppm > 400:
            eb += 0.4
            reasons.append(f"V-shaped profile (fill {f.sharpness:.2f})")
        if f.odd_even_ppm > c.odd_even_ratio * f.depth_ppm:
            eb += 0.3
            reasons.append("odd-even depth mismatch")
        if eb >= 0.5:
            return Prediction("eclipse", float(min(0.55 + eb / 2, 0.98)), reasons, "rule")

        reasons = [f"U-shaped achromatic dip (fill {f.sharpness:.2f})",
                   f"SNR {f.snr:.1f}, depth {f.depth_ppm:.0f} ppm",
                   "no secondary eclipse; centroid on-target"]
        conf = float(np.clip(0.7 + (f.snr - c.snr_min) * 0.006, 0.7, 0.985))
        return Prediction("transit", conf, reasons, "rule")


@register_classifier("sklearn")
class SklearnClassifier(BaseClassifier):
    """Gradient-boosted trees on physical features; trainable + serialisable."""

    def __init__(self, model=None):
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnClassifier":
        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(random_state=42)
        self.model.fit(X, y)
        return self

    def predict_one(self, f: TransitFeatures) -> Prediction:
        if self.model is None:
            raise RuntimeError("SklearnClassifier is not trained; call fit() or load().")
        x = f.to_vector().reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        classes = list(self.model.classes_)
        i = int(np.argmax(proba))
        label = classes[i]
        reasons = [f"model probability {proba[i]:.2f}",
                   f"sharpness {f.sharpness:.2f}, SNR {f.snr:.1f}",
                   f"depth {f.depth_ppm:.0f} ppm"]
        return Prediction(label, float(proba[i]), reasons, "sklearn")

    def save(self, path: str | Path) -> None:
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnClassifier":
        import joblib
        return cls(model=joblib.load(path))


@register_classifier("hybrid")
class HybridClassifier(BaseClassifier):
    """Production default: a physics detection-gate + a real-data-trained ML decision.

    The 'is there even a transit-like signal?' step (SNR / localization / width) stays a
    transparent physics gate (the ML has no starspot training). When a real signal is
    present, the **ML trained on real TESS data** decides planet vs eclipsing-binary — where
    cross-validation showed it beats the rules (0.95 vs 0.75).
    """

    def __init__(self, ml: SklearnClassifier, config: ClassifyConfig | None = None):
        self.ml = ml
        self.cfg = config or ClassifyConfig()
        self.rule = RuleBasedClassifier(self.cfg)

    def predict_one(self, f: TransitFeatures) -> Prediction:
        c = self.cfg
        if (f.snr < c.snr_min or f.depth_ppm < c.depth_min_ppm
                or not f.localized or f.duration_phase > c.max_duration_phase):
            return self.rule.predict_one(f)            # starspot / no convincing transit
        p = self.ml.predict_one(f)
        reasons = [f"ML (real-trained) probability {p.confidence:.2f}", *p.reasons[1:]]
        return Prediction(p.label, p.confidence, reasons, "hybrid-ml")


def build_training_set(n_per_class: int = 60, seed: int = 0):
    """Generate synthetic light curves across classes and extract their features."""
    from .data import from_synthetic
    from .pipeline import features_from_lightcurve

    rng = np.random.default_rng(seed)
    X, y = [], []
    grids = {  # depths deliberately overlap so the model learns shape/secondary, not depth
        "transit": dict(kind="planet", depth=(300, 12000), dur=(1.0, 4.5), period=(0.7, 9.0)),
        "eclipse": dict(kind="eb", depth=(5000, 25000), dur=(1.5, 5.0), period=(0.8, 6.0)),
        "starspot": dict(kind="starspot", depth=(0, 0), dur=(2.0, 4.0), period=(1.0, 8.0)),
    }
    for label, g in grids.items():
        for k in range(n_per_class):
            lc = from_synthetic(
                kind=g["kind"],
                period_days=float(rng.uniform(*g["period"])),
                depth_ppm=float(rng.uniform(*g["depth"])),
                duration_hours=float(rng.uniform(*g["dur"])),
                noise_ppm=float(rng.uniform(150, 450)),
                variability_ppm=float(rng.uniform(800, 4500)),
                seed=int(rng.integers(0, 1_000_000)),
            )
            feats = features_from_lightcurve(lc)
            X.append(feats.to_vector())
            y.append(label)
    return np.asarray(X), np.asarray(y)


def train_default_model(n_per_class: int = 60, seed: int = 0) -> SklearnClassifier:
    X, y = build_training_set(n_per_class=n_per_class, seed=seed)
    return SklearnClassifier().fit(X, y)
