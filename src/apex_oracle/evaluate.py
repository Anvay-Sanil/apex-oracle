"""Validation: injection-recovery, classification metrics, calibration.

These are the numbers that earn scientific trust (Tanaka's mandate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig
from .data import LightCurve, from_synthetic
from .features import TransitFeatures
from .model import BaseClassifier, RuleBasedClassifier
from .pipeline import ExoplanetPipeline, features_from_lightcurve

_CLASS_KIND = {"transit": "planet", "eclipse": "eb", "starspot": "starspot"}


def _gen(label: str, rng: np.random.Generator) -> LightCurve:
    if label == "transit":
        depth, dur, per = rng.uniform(300, 4000), rng.uniform(1.5, 4.0), rng.uniform(1.5, 8.0)
    elif label == "eclipse":
        depth, dur, per = rng.uniform(6000, 25000), rng.uniform(2.0, 5.0), rng.uniform(1.2, 6.0)
    else:
        depth, dur, per = 0.0, rng.uniform(2.0, 4.0), rng.uniform(2.0, 8.0)
    return from_synthetic(
        kind=_CLASS_KIND[label], period_days=float(per), depth_ppm=float(depth),
        duration_hours=float(dur), noise_ppm=float(rng.uniform(150, 450)),
        variability_ppm=float(rng.uniform(800, 4500)), seed=int(rng.integers(0, 1_000_000)),
    )


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    per_class: dict          # label -> {precision, recall, f1, support}
    confusion: dict          # "true->pred" counts
    ece: float
    n: int


def evaluate_classifier(
    classifier: BaseClassifier | None = None,
    n_per_class: int = 40,
    seed: int = 123,
    config: PipelineConfig | None = None,
) -> ClassificationReport:
    """Run a fresh synthetic test set through the full pipeline and score it."""
    clf = classifier or RuleBasedClassifier()
    pipe = ExoplanetPipeline(config, classifier=clf)
    rng = np.random.default_rng(seed)
    labels = ("transit", "eclipse", "starspot")

    y_true, y_pred, conf = [], [], []
    for label in labels:
        for _ in range(n_per_class):
            res = pipe.run(_gen(label, rng))
            y_true.append(label)
            y_pred.append(res.prediction.label)
            conf.append(res.prediction.confidence)

    y_true, y_pred, conf = np.array(y_true), np.array(y_pred), np.array(conf)
    per_class = {}
    for label in labels:
        tp = int(np.sum((y_pred == label) & (y_true == label)))
        fp = int(np.sum((y_pred == label) & (y_true != label)))
        fn = int(np.sum((y_pred != label) & (y_true == label)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[label] = {"precision": prec, "recall": rec, "f1": f1,
                            "support": int(np.sum(y_true == label))}
    confusion = {}
    for tl in labels:
        for pl in labels:
            confusion[f"{tl}->{pl}"] = int(np.sum((y_true == tl) & (y_pred == pl)))
    accuracy = float(np.mean(y_true == y_pred))
    ece = expected_calibration_error(conf, y_true == y_pred)
    return ClassificationReport(accuracy, per_class, confusion, ece, len(y_true))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Standard ECE over equal-width confidence bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def injection_recovery(
    depth_grid_ppm=(200, 400, 800, 1500, 3000),
    n_per_cell: int = 12,
    seed: int = 7,
    config: PipelineConfig | None = None,
) -> dict:
    """Inject planets of varying depth into noisy stars; measure recovery completeness."""
    pipe = ExoplanetPipeline(config, classifier=RuleBasedClassifier())  # synthetic detection test
    rng = np.random.default_rng(seed)
    out = {}
    for depth in depth_grid_ppm:
        recovered = 0
        for _ in range(n_per_cell):
            period = float(rng.uniform(2.0, 6.0))
            lc = from_synthetic(
                kind="planet", period_days=period, depth_ppm=float(depth),
                duration_hours=float(rng.uniform(2.0, 3.5)),
                noise_ppm=float(rng.uniform(200, 400)),
                variability_ppm=float(rng.uniform(800, 2500)),
                seed=int(rng.integers(0, 1_000_000)),
            )
            res = pipe.run(lc)
            if res.prediction.label == "transit":
                recovered += 1
        out[int(depth)] = recovered / n_per_cell
    return out
