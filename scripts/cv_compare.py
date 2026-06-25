"""Leakage-free cross-validated head-to-head on REAL labelled TESS features.

Reads data/real_features.csv (built by build_real_dataset.py) and compares:
  - rule-based physics classifier (no training)
  - ML (gradient-boosted trees) via stratified cross-validation (each target = 1 sample,
    so 1 sample per star => no per-star leakage; CV gives an honest out-of-sample score).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from apex_oracle.config import ClassifyConfig
from apex_oracle.features import TransitFeatures
from apex_oracle.model import RuleBasedClassifier

CSV = Path("data/real_features.csv")


def _row_to_features(row: dict) -> TransitFeatures:
    depth = float(row["depth_ppm"])
    return TransitFeatures(
        period_days=float(row["period_days"]), depth_ppm=depth,
        duration_hours=float(row["duration_h"]), snr=float(row["snr"]),
        sharpness=float(row["sharpness"]),
        secondary_ppm=float(row["secondary_ratio"]) * depth,
        odd_even_ppm=float(row["odd_even_ratio"]) * depth,
        oot_rms_ppm=float(row["oot_rms_ppm"]), duration_phase=float(row["dur_phase"]),
        localized=depth > 2.5 * float(row["oot_rms_ppm"]),
        depth_err_ppm=0.0, period_err_days=0.0, duration_err_hours=0.0,
    )


def _metrics(name, y, pred, labels):
    acc = float(np.mean(y == pred))
    print(f"\n{name}: accuracy {acc:.3f}")
    for lab in labels:
        sub = y == lab
        if sub.sum():
            rec = float(np.mean(pred[sub] == lab))
            tp = int(np.sum((pred == lab) & sub)); fp = int(np.sum((pred == lab) & ~sub))
            prec = tp / (tp + fp) if tp + fp else 0.0
            print(f"   {lab:9s} recall {rec:.2f}  precision {prec:.2f}  (n={int(sub.sum())})")
    return acc


def main() -> int:
    if not CSV.exists():
        print("run build_real_dataset.py first"); return 1
    rows = list(csv.DictReader(CSV.open()))
    X = np.array([[float(r[c]) for c in TransitFeatures.vector_names()] for r in rows])
    y = np.array([r["label"] for r in rows])
    labels = sorted(set(y))
    print(f"dataset: {len(rows)} real targets | " +
          " ".join(f"{l}={int(np.sum(y==l))}" for l in labels))

    # rule-based (no training)
    rule = RuleBasedClassifier(ClassifyConfig())
    rule_pred = np.array([rule.predict_one(_row_to_features(r)).label for r in rows])
    _metrics("RULE-BASED (physics, no training)", y, rule_pred, labels)

    # ML with stratified CV (out-of-sample predictions)
    n_splits = min(5, int(np.min([np.sum(y == l) for l in labels])))
    if n_splits < 2:
        print("\nnot enough per-class samples for CV"); return 0
    clf = GradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    ml_pred = cross_val_predict(clf, X, y, cv=cv)
    _metrics(f"ML real-trained (gradient boosting, {n_splits}-fold CV)", y, ml_pred, labels)

    # ---- headline: binary planet vs false-positive (the vetting task) ----
    yb = np.where(y == "transit", "planet", "fp")
    rule_b = np.where(rule_pred == "transit", "planet", "fp")
    print("\n================ BINARY: planet vs false-positive ================")
    _metrics("RULE-BASED (physics, no training)", yb, rule_b, ["planet", "fp"])
    nb = min(5, int(np.min([np.sum(yb == l) for l in ("planet", "fp")])))
    if nb >= 2:
        cvb = StratifiedKFold(n_splits=nb, shuffle=True, random_state=42)
        ml_b = cross_val_predict(GradientBoostingClassifier(random_state=42), X, yb, cv=cvb)
        _metrics(f"ML real-trained ({nb}-fold CV)", yb, ml_b, ["planet", "fp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
