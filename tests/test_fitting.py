"""Transit light-curve fitting tests (no network)."""
from __future__ import annotations

from apex_oracle import ExoplanetPipeline
from apex_oracle.data import from_synthetic
from apex_oracle.vetting import BlendResult


def test_fit_gives_parameters_with_errors():
    res = ExoplanetPipeline().run(from_synthetic(kind="planet", seed=5))
    f = res.fit
    assert f is not None
    assert f.converged
    assert f.depth_ppm > 0 and f.depth_err_ppm > 0          # fitted depth + 1-sigma error
    assert f.duration_hours > 0 and f.duration_err_hours >= 0
    assert f.reduced_chi2 > 0
    # the pipeline reports fitted depth + reduced chi-squared
    assert res.parameters["depth_ppm"][0] > 0
    assert "reduced_chi2" in res.parameters


def test_blend_result_shape():
    # offline / unresolved target -> graceful BlendResult (never raises)
    br = BlendResult(False, 0.0, 0.0, False, "n/a")
    d = br.as_dict()
    assert set(d) >= {"available", "is_blend", "offset_arcsec", "significance_sigma"}
    assert isinstance(br.is_blend, bool)
