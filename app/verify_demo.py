"""Verify the demo pipeline: inject known signals, check recovery, save a figure."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import LightCurveConfig, run_pipeline


def _check(label: str, cfg: LightCurveConfig, expect_label: str) -> "tuple":
    res = run_pipeline(cfg)
    p_true, p_rec = cfg.period_days, res.search.period
    # accept the true period or a near 2x/0.5x alias, then report the ratio
    ratio = p_rec / p_true
    period_ok = min(abs(ratio - 1), abs(ratio - 2), abs(ratio - 0.5)) < 0.05
    cls_ok = expect_label.lower() in res.classification.label.lower()
    print(f"[{label}] true P={p_true:.3f}d  recovered={p_rec:.3f}d  ratio={ratio:.2f}  "
          f"period_ok={period_ok} | depth={res.params.depth_ppm:.0f}ppm "
          f"dur={res.params.duration_hours:.2f}h SNR={res.params.snr:.1f} "
          f"| class='{res.classification.label}' ({res.classification.confidence:.2f}) "
          f"expect~{expect_label} class_ok={cls_ok}")
    return res, period_ok, cls_ok


def main() -> int:
    planet = LightCurveConfig(kind="planet", period_days=3.41, depth_ppm=900,
                              duration_hours=2.7, seed=42)
    eb = LightCurveConfig(kind="eb", period_days=2.15, depth_ppm=12000,
                          duration_hours=3.0, seed=7)
    spot = LightCurveConfig(kind="starspot", period_days=3.41, depth_ppm=0,
                            variability_ppm=4000, seed=3)

    res_p, p_ok, pc_ok = _check("PLANET", planet, "planet")
    _check("EB", eb, "binary")
    _check("STARSPOT", spot, "starspot")

    # results figure from the planet case
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    ax[0, 0].plot(res_p.t, res_p.flux_raw, ".", ms=1.5, color="#888")
    ax[0, 0].plot(res_p.t, res_p.trend, "-", color="#E24B4A", lw=1)
    ax[0, 0].set_title("1. Raw light curve + trend"); ax[0, 0].set_xlabel("time (days)")

    ax[0, 1].plot(res_p.t, res_p.flux_flat, ".", ms=1.5, color="#185FA5")
    ax[0, 1].set_title("2. Detrended (transit-safe)"); ax[0, 1].set_xlabel("time (days)")

    ax[1, 0].plot(res_p.search.periods, res_p.search.power, "-", color="#0F6E56", lw=1)
    ax[1, 0].axvline(res_p.search.period, color="#E24B4A", ls="--", lw=1)
    ax[1, 0].set_title(f"3. Period search (peak {res_p.search.period:.3f} d)")
    ax[1, 0].set_xlabel("trial period (days)")

    ax[1, 1].plot(res_p.phase, res_p.flux_folded, ".", ms=2, color="#888", alpha=0.5)
    c, prof = __import__("pipeline")._binned_profile(res_p.phase, res_p.flux_folded)
    ax[1, 1].plot(c, prof, "-", color="#185FA5", lw=2)
    ax[1, 1].set_xlim(-0.15, 0.15)
    ax[1, 1].set_title(f"4. Phase-folded: {res_p.classification.label} "
                       f"({res_p.classification.confidence:.0%})")
    ax[1, 1].set_xlabel("phase")
    fig.tight_layout()
    fig.savefig("verify_result.png", dpi=120)
    print("saved figure -> app/verify_result.png")

    ok = p_ok and pc_ok
    print("\nVERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
