"""APEX-ORACLE - Exoplanet Inspector (Streamlit demo).

Run with:  streamlit run app/dashboard.py
Self-contained synthetic-data demo of the detection -> classification ->
parameter-estimation pipeline with explainable, phase-folded visuals.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pipeline import LightCurveConfig, _binned_profile, attention_weights, run_pipeline

st.set_page_config(page_title="APEX-ORACLE Exoplanet Inspector", layout="wide")

PRESETS: dict[str, dict] = {
    "Planet (hot Neptune)": dict(kind="planet", period_days=3.41, depth_ppm=900,
                                 duration_hours=2.7, noise_ppm=250, variability_ppm=1500),
    "Eclipsing binary": dict(kind="eb", period_days=2.15, depth_ppm=12000,
                             duration_hours=3.0, noise_ppm=300, variability_ppm=1200),
    "Starspot / variable star": dict(kind="starspot", period_days=3.40, depth_ppm=0,
                                     duration_hours=2.5, noise_ppm=300, variability_ppm=4000),
}

st.title("APEX-ORACLE  ·  Exoplanet Inspector")
st.caption("Physics-aware, explainable transit detection — demo on synthetic TESS-like data.")

# --------------------------- controls --------------------------- #
sb = st.sidebar
sb.header("Scenario")
preset = sb.selectbox("Preset", list(PRESETS) + ["Custom"])
base = PRESETS.get(preset, PRESETS["Planet (hot Neptune)"])

if preset == "Custom":
    kind = sb.selectbox("Signal kind", ["planet", "eb", "starspot"])
    period = sb.slider("Period (days)", 0.5, 10.0, 3.41, 0.01)
    depth = sb.slider("Depth (ppm)", 0, 30000, 900, 50)
    duration = sb.slider("Duration (hours)", 0.5, 6.0, 2.7, 0.1)
    noise = sb.slider("White noise (ppm)", 50, 1000, 250, 10)
    variability = sb.slider("Stellar variability (ppm)", 0, 6000, 1500, 100)
else:
    kind = base["kind"]
    period, depth = base["period_days"], base["depth_ppm"]
    duration, noise = base["duration_hours"], base["noise_ppm"]
    variability = sb.slider("Stellar variability (ppm)", 0, 6000, base["variability_ppm"], 100)
    noise = sb.slider("White noise (ppm)", 50, 1000, base["noise_ppm"], 10)

seed = int(sb.number_input("Random seed", 0, 9999, 42))
sb.markdown("---")
sb.caption("The classifier shown is a transparent rule-based stand-in for the trained "
           "multimodal Transformer, using the same physical features.")

cfg = LightCurveConfig(period_days=period, depth_ppm=depth, duration_hours=duration,
                       noise_ppm=noise, variability_ppm=variability, kind=kind, seed=seed)
res = run_pipeline(cfg)
p, cls = res.params, res.classification

# --------------------------- verdict + metrics --------------------------- #
left, right = st.columns([5, 4])
with left:
    if cls.label.startswith("Planet"):
        st.success(f"### {cls.label}\nconfidence {cls.confidence:.0%}")
    elif cls.label.startswith("Eclipsing"):
        st.warning(f"### {cls.label}\nconfidence {cls.confidence:.0%}")
    else:
        st.info(f"### {cls.label}\nconfidence {cls.confidence:.0%}")
    st.markdown("**Why:**")
    for r in cls.reasons:
        st.markdown(f"- {r}")

with right:
    c1, c2 = st.columns(2)
    c1.metric("Period (days)", f"{p.period_days:.3f}")
    c2.metric("Depth (ppm)", f"{p.depth_ppm:,.0f}")
    c3, c4 = st.columns(2)
    c3.metric("Duration (hours)", f"{p.duration_hours:.2f}")
    c4.metric("SNR", f"{p.snr:.1f}")
    st.caption(f"secondary {p.secondary_ppm:,.0f} ppm · odd-even "
               f"{p.odd_even_diff_ppm:,.0f} ppm · flatness {p.flatness:.2f}")

st.markdown("---")

# --------------------------- plots --------------------------- #
GRAY, BLUE, RED, TEAL = "#888888", "#185FA5", "#E24B4A", "#0F6E56"
a, b = st.columns(2)
with a:
    fig1, ax1 = plt.subplots(figsize=(5.4, 2.7))
    ax1.plot(res.t, res.flux_raw, ".", ms=1.5, color=GRAY)
    ax1.plot(res.t, res.trend, "-", color=RED, lw=1)
    ax1.set_title("Raw light curve + fitted trend"); ax1.set_xlabel("time (days)")
    fig1.tight_layout(); st.pyplot(fig1)

    fig3, ax3 = plt.subplots(figsize=(5.4, 2.7))
    ax3.plot(res.search.periods, res.search.power, "-", color=TEAL, lw=0.9)
    ax3.axvline(res.search.period, color=RED, ls="--", lw=1)
    ax3.set_title(f"Period search (peak {res.search.period:.3f} d)")
    ax3.set_xlabel("trial period (days)")
    fig3.tight_layout(); st.pyplot(fig3)

with b:
    fig2, ax2 = plt.subplots(figsize=(5.4, 2.7))
    ax2.plot(res.t, res.flux_flat, ".", ms=1.5, color=BLUE)
    ax2.set_title("Detrended (transit-safe)"); ax2.set_xlabel("time (days)")
    fig2.tight_layout(); st.pyplot(fig2)

    fig4, ax4 = plt.subplots(figsize=(5.4, 2.7))
    ax4.plot(res.phase, res.flux_folded, ".", ms=2, color=GRAY, alpha=0.45)
    centers, prof = _binned_profile(res.phase, res.flux_folded)
    ax4.plot(centers, prof, "-", color=BLUE, lw=2)
    # explainability: shade where the model "attends" (in-transit)
    dur_phase = max(p.duration_hours / 24.0 / max(p.period_days, 1e-6), 1e-3)
    w = attention_weights(centers, dur_phase)
    ax4.fill_between(centers, prof.min(), prof.max(), where=w > 0.3,
                     color=RED, alpha=0.12, label="model attention")
    ax4.set_xlim(-0.15, 0.15); ax4.legend(loc="lower right", fontsize=8)
    ax4.set_title(f"Phase-folded + attention: {cls.label}")
    ax4.set_xlabel("phase")
    fig4.tight_layout(); st.pyplot(fig4)

st.caption("Demo pipeline: generate → detrend → BLS-style search → phase fold → "
           "estimate (period, depth, duration, SNR) → classify. "
           "Next: swap the rule-based classifier for the trained Transformer and "
           "ingest real TESS light curves via lightkurve.")
