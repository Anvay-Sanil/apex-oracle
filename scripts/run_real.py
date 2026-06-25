"""Download a real TESS target, run APEX-ORACLE, and save a phase-folded figure.

Usage:  python scripts/run_real.py "WASP-18"
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apex_oracle import ExoplanetPipeline
from apex_oracle.data import from_tess
from apex_oracle.preprocess import binned_profile


def main(target: str) -> int:
    print(f"[1/3] downloading TESS light curve for {target} ...")
    lc = from_tess(target)
    print(f"      {lc.time.size} cadences, source={lc.source}")

    print("[2/3] running pipeline ...")
    res = ExoplanetPipeline().run(lc)
    print("\n" + res.summary())
    print(f"      classification: {res.prediction.label.upper()} "
          f"({res.prediction.confidence:.0%})")
    for r in res.prediction.reasons:
        print(f"        - {r}")

    print("[3/3] saving figure ...")
    out = Path("outputs"); out.mkdir(exist_ok=True)
    phase, folded = res.arrays["phase"], res.arrays["folded"]
    centers, prof = binned_profile(phase, folded, 240)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(res.arrays["time"], res.arrays["flat"], ".", ms=1, color="#185FA5")
    ax[0].set_title(f"{target}: detrended TESS light curve"); ax[0].set_xlabel("time (d)")
    ax[1].plot(phase, folded, ".", ms=1.5, color="#888", alpha=0.4)
    ax[1].plot(centers, prof, "-", color="#0F6E56", lw=2)
    ax[1].set_xlim(-0.12, 0.12)
    ax[1].set_title(f"phase-folded: {res.prediction.label} "
                    f"(P={res.period_days:.3f} d)")
    ax[1].set_xlabel("phase")
    fig.tight_layout()
    path = out / f"real_{target.replace(' ', '_')}.png"
    fig.savefig(path, dpi=120)
    print(f"      saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "WASP-18"))
