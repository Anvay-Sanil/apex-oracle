"""Blend vetting via Target Pixel File centroids.

A *blend* means the brightness dip comes from a neighbouring/background source inside the
aperture, not the target star. During such a dip the flux-weighted photocentre shifts toward
that source. We compare the in-transit vs out-of-transit centroid; a significant shift flags a
blend. This needs pixel-level data (a TESS Target Pixel File) plus the transit ephemeris, so it
is a real-data step (the [astro] extra) and not applicable to a bare 1-D light curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TESS_PIXEL_ARCSEC = 21.0


@dataclass(frozen=True)
class BlendResult:
    available: bool
    offset_arcsec: float
    significance: float
    is_blend: bool
    note: str

    def as_dict(self) -> dict:
        return {"available": self.available, "offset_arcsec": round(self.offset_arcsec, 3),
                "significance_sigma": round(self.significance, 2), "is_blend": self.is_blend,
                "note": self.note}


def blend_test(target: str, period_days: float, t0_days: float, duration_hours: float,
               sig_threshold: float = 3.0, offset_threshold_arcsec: float = 2.0) -> BlendResult:
    """Centroid-shift blend test on a real TESS target. Returns a BlendResult (never raises)."""
    try:
        import lightkurve as lk
    except ImportError:
        return BlendResult(False, 0.0, 0.0, False, "needs the [astro] extra (lightkurve)")
    try:
        sr = lk.search_targetpixelfile(target, mission="TESS", author="SPOC", exptime=120)
        if len(sr) == 0:
            return BlendResult(False, 0.0, 0.0, False, f"no SPOC target-pixel file for {target}")
        tpf = sr.download()
        col, row = tpf.estimate_centroids()
        t = np.asarray(tpf.time.value, dtype=float)
        col = np.asarray(col, dtype=float)
        row = np.asarray(row, dtype=float)
        ok = np.isfinite(t) & np.isfinite(col) & np.isfinite(row)
        t, col, row = t[ok], col[ok], row[ok]

        dur = duration_hours / 24.0
        ph = (t - t0_days) % period_days
        in_tr = (ph < dur / 2) | (ph > period_days - dur / 2)
        oot = ~in_tr
        if int(in_tr.sum()) < 5 or int(oot.sum()) < 20:
            return BlendResult(False, 0.0, 0.0, False, "too few in/out-of-transit cadences")

        d_col = float(col[in_tr].mean() - col[oot].mean())
        d_row = float(row[in_tr].mean() - row[oot].mean())
        offset_px = float(np.hypot(d_col, d_row))
        scatter = float(np.hypot(np.std(col[oot]), np.std(row[oot])) / np.sqrt(int(in_tr.sum())))
        sig = offset_px / scatter if scatter > 0 else 0.0
        offset_arcsec = offset_px * TESS_PIXEL_ARCSEC
        is_blend = sig >= sig_threshold and offset_arcsec >= offset_threshold_arcsec
        verdict = "BLEND (off-target dip)" if is_blend else "on-target (no significant shift)"
        return BlendResult(True, offset_arcsec, sig, is_blend,
                           f"{verdict}: centroid offset {offset_arcsec:.2f}\" ({sig:.1f} sigma)")
    except Exception as exc:  # network / data hiccup -> non-fatal
        return BlendResult(False, 0.0, 0.0, False, f"blend test unavailable: {type(exc).__name__}")
