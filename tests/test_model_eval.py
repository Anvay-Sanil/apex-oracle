"""Training, model serialisation and validation-harness tests."""
from __future__ import annotations

from apex_oracle.data import from_synthetic
from apex_oracle.evaluate import (
    evaluate_classifier,
    expected_calibration_error,
    injection_recovery,
)
from apex_oracle.model import LABELS, SklearnClassifier, train_default_model
from apex_oracle.pipeline import features_from_lightcurve
import numpy as np


def test_train_predict_and_serialise(tmp_path):
    clf = train_default_model(n_per_class=20, seed=0)
    feats = features_from_lightcurve(from_synthetic(kind="planet", seed=2))
    pred = clf.predict_one(feats)
    assert pred.label in LABELS
    assert 0.0 <= pred.confidence <= 1.0

    path = tmp_path / "m.joblib"
    clf.save(path)
    reloaded = SklearnClassifier.load(path)
    assert reloaded.predict_one(feats).label in LABELS


def test_evaluate_classifier_reasonable():
    rep = evaluate_classifier(n_per_class=8, seed=321)
    assert rep.n == 24
    assert rep.accuracy > 0.6          # rule-based baseline should be well above chance
    assert 0.0 <= rep.ece <= 1.0
    assert set(rep.per_class) == set(LABELS)


def test_injection_recovery_monotone_ish():
    inj = injection_recovery(n_per_cell=6, seed=42)
    assert inj[3000] >= 0.5             # deep transits are reliably recovered
    assert inj[3000] >= inj[200]        # deeper -> not worse than shallow


def test_ece_bounds():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 1])
    assert 0.0 <= expected_calibration_error(conf, correct) <= 1.0
