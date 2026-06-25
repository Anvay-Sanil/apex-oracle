"""Train the production classifier on REAL labelled TESS features and bundle it.

Reads data/real_features.csv and saves the trained model to
src/apex_oracle/models/default.joblib (loaded as the pipeline default).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from apex_oracle.features import TransitFeatures
from apex_oracle.model import SklearnClassifier

CSV = Path("data/real_features.csv")
OUT = Path("src/apex_oracle/models/default.joblib")


def main() -> int:
    rows = list(csv.DictReader(CSV.open()))
    X = np.array([[float(r[c]) for c in TransitFeatures.vector_names()] for r in rows])
    y = np.array([r["label"] for r in rows])
    clf = SklearnClassifier()
    clf.model = GradientBoostingClassifier(random_state=42).fit(X, y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    clf.save(OUT)
    print(f"trained on {len(rows)} real targets; classes {list(clf.model.classes_)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
