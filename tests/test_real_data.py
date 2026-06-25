"""Real-TESS integration test. Skipped automatically when offline or without lightkurve."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.network


def _online() -> bool:
    try:
        import lightkurve  # noqa: F401
    except ImportError:
        return False
    import urllib.request
    try:
        urllib.request.urlopen("https://mast.stsci.edu", timeout=8)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _online(), reason="MAST / lightkurve unavailable")
@pytest.mark.parametrize("target", ["WASP-18", "TIC 100100827"])
def test_real_tess_ingestion(target):
    from apex_oracle import ExoplanetPipeline
    from apex_oracle.data import from_tess

    try:
        lc = from_tess(target)
    except Exception as exc:  # network hiccup / target moved
        pytest.skip(f"could not fetch {target}: {exc}")
    assert lc.time.size > 100
    res = ExoplanetPipeline().run(lc)
    assert res.prediction.label in {"transit", "eclipse", "starspot"}
    assert res.features.period_days > 0
