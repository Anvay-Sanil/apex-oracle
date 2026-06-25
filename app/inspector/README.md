# APEX-ORACLE — Exoplanet Inspector (demo)

A standalone, browser-based mission-control demo of the APEX-ORACLE pipeline:
detect → detrend → BLS period search → phase-fold → estimate (period, depth, duration, SNR)
→ classify (planet transit / eclipsing binary / starspot), with an explainable phase-folded
view and a 3D star–planet transit visualization (Three.js / WebGL).

## Run
Open `index.html` in a modern browser (Chrome/Edge/Firefox). No build, no backend.
Or serve it: `python -m http.server 8530 --directory app/inspector` then open
http://localhost:8530. Internet is needed once to load Three.js + fonts from CDN. If WebGL is
unavailable, the charts and analysis still run.

## Two modes
- **Simulation (default):** synthetic TESS-like light curves you control with the sidebar
  (scenario presets + sliders). Clearly labelled `◈ SIMULATION / SOURCE SYNTHETIC`.
- **Real data:** click **LOAD REAL CSV** and upload a light curve (CSV/TXT, two columns
  `time, flux`; header lines and `#` comments are skipped). The same pipeline runs on your
  data and the header switches to `◉ REAL DATA`.
- **Real engine** (`◆ REAL ENGINE`): start the validated backend with `apex-oracle serve`,
  type a target (a TIC id, a confirmed-planet name like `WASP-18`, or `synthetic:planet`) and
  click **ANALYZE VIA ENGINE**. The UI calls `POST /analyze` on the Python service, which
  downloads real TESS data and runs the *same validated pipeline* used by the tests and the
  benchmark — so the demo and the engine are one system, not two implementations.

## What is real vs. a stand-in (honesty)
Real, in-browser: detrending, BLS-style period search (with 2×/3× de-aliasing), phase folding,
robust median-based depth/duration/SNR with uncertainties, and a transparent rule-based
classifier (U-vs-V sharpness, secondary eclipse, dip localization/width, SNR).

Stand-in for the full system: the classifier is a rule-based proxy for the trained multimodal
Transformer; parameters are not yet full Bayesian posteriors; there is no GP detrend, no
centroid / difference-imaging vetting, and no injection-recovery metrics yet.

## Known limitations
- BLS can still pick a period alias on sparse or low-SNR data.
- The 3D view is illustrative: the plotted dip depth is quantitative, but the planet's
  on-screen size is exaggerated for visibility.
- Not yet validated on real labelled TESS data — see roadmap.

## Roadmap to production (ISRO-grade)
- Real ingestion via `lightkurve`; GP detrend (`celerite2`) + `transitleastsquares`.
- Trained classifier with injection-recovery completeness/reliability + calibration.
- Differentiable `batman` + SBI posteriors; centroid / difference-imaging vetting.
- Tests, CI, packaging, deployment, accessibility.

## Verification
Planet / eclipsing binary / starspot all classify correctly in both simulation and real-CSV
modes (browser-verified, 0 console errors). See `../../task_plan.md`.
