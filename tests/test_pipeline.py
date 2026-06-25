"""End-to-end pipeline tests on synthetic signals."""
from __future__ import annotations

import numpy as np
import pytest

from apex_oracle import ExoplanetPipeline, load_lightcurve
from apex_oracle.data import from_csv, from_synthetic
from apex_oracle.model import HybridClassifier, RuleBasedClassifier


def _period_ok(rec: float, true: float, tol: float = 0.1) -> bool:
    ratio = rec / true
    return min(abs(ratio - 1), abs(ratio - 2), abs(ratio - 0.5)) < tol


@pytest.mark.parametrize("kind,expected", [
    ("planet", "transit"),
    ("eb", "eclipse"),
    ("starspot", "starspot"),
])
def test_classification_rule_path(kind, expected):
    # Synthetic signals exercise the transparent physics classifier. The real-data-trained
    # ML default is validated separately on real TESS data (scripts/cv_compare.py: 0.95 CV).
    lc = from_synthetic(kind=kind, seed=11)
    res = ExoplanetPipeline(classifier=RuleBasedClassifier()).run(lc)
    assert res.prediction.label == expected
    assert 0.0 <= res.prediction.confidence <= 1.0


def test_default_is_real_trained_hybrid():
    p = ExoplanetPipeline()
    assert isinstance(p.classifier, HybridClassifier)        # ships the real-trained ML default
    res = p.run(from_synthetic(kind="planet", seed=1))
    assert res.prediction.label in {"transit", "eclipse", "starspot"}


def test_planet_period_recovered():
    lc = from_synthetic(kind="planet", period_days=3.41, seed=3)
    res = ExoplanetPipeline().run(lc)
    assert _period_ok(res.period_days, 3.41)


def test_parameters_have_uncertainties():
    res = ExoplanetPipeline().run(from_synthetic(kind="planet", seed=5))
    f = res.features
    assert f.depth_err_ppm > 0
    assert f.period_err_days > 0
    assert f.duration_err_hours > 0
    assert res.parameters["depth_ppm"][0] > 0


def test_csv_roundtrip(tmp_path):
    lc = from_synthetic(kind="planet", seed=9)
    p = tmp_path / "lc.csv"
    np.savetxt(p, np.column_stack([lc.time, lc.flux]), delimiter=",", header="time,flux")
    loaded = from_csv(p)
    assert loaded.time.size > 1000
    res = load_lightcurve(str(p))  # dispatch path
    assert res.source.startswith("csv:")
    out = ExoplanetPipeline().run(from_csv(p))
    assert out.prediction.label in {"transit", "eclipse", "starspot"}


def test_vetting_flags_present():
    res = ExoplanetPipeline().run(from_synthetic(kind="eb", seed=7))
    assert "secondary_flag" in res.vetting
    assert "centroid" in res.vetting
