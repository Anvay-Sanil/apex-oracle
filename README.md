# APEX-ORACLE

**Explainable AI detection, classification and characterisation of exoplanet transits in
noisy TESS light curves.** Built for the Bharatiya Antariksh Hackathon 2026 (ISRO).

APEX-ORACLE is a packaged Python library + CLI (and a browser front-end) that takes a light
curve and returns: a **detection**, a **classification** (transit / eclipsing binary /
starspot) with reasons, **physical parameters with uncertainties** (period, depth, duration,
SNR), and **vetting flags** — with the validation numbers that make those answers trustworthy.

> **Honesty first.** This repository is a *real, tested engineering foundation*, not a
> finished, certified ISRO system. It ships a trained baseline and an injection-recovery
> harness. Production deployment in an ISRO lab additionally requires validation on their
> labelled data, a security review, and scale training — see [Roadmap](#roadmap).

---

## Install

```bash
uv venv && uv pip install -e .            # core (synthetic + CSV, no heavy astro deps)
uv pip install -e ".[astro]"              # + real TESS via lightkurve, wotan, TLS
uv pip install -e ".[dev]"                # + pytest / ruff / mypy
```

(Plain `pip install -e .` works too.)

## Quickstart — CLI

```bash
apex-oracle demo --kind planet            # run on a synthetic planet
apex-oracle run path/to/lightcurve.csv    # run on a real CSV (columns: time, flux)
apex-oracle run "TIC 307210830"           # real TESS download (needs [astro] extra)
apex-oracle train --out model.joblib      # train the classifier on synthetic injections
apex-oracle evaluate                      # classification metrics + injection-recovery
apex-oracle serve --port 8800             # run the JSON API service
```

## Service — JSON API

The validated engine deploys as a zero-dependency HTTP service (the integration point for a
UI or batch jobs):

```bash
apex-oracle serve --port 8800
curl localhost:8800/health
curl -X POST localhost:8800/analyze -d '{"source":"synthetic:planet"}'
# -> {"classification":"transit","confidence":0.81,
#     "parameters":{"period_days":[3.414,0.014],"depth_ppm":[667,26], ...},
#     "reasons":[...], "vetting":{...}, "folded_curve":{...}}
```

`POST /analyze` accepts `{"source": "..."}` (synthetic / CSV path / `TIC ...`) or
`{"csv": "<time,flux text>"}`. The container (`Dockerfile`) and `apex-oracle serve` make it
deployable as a microservice.

## Quickstart — Python API

```python
from apex_oracle import ExoplanetPipeline, load_lightcurve

lc = load_lightcurve("synthetic:planet")        # or a CSV path, or "TIC ..."
result = ExoplanetPipeline().run(lc)
print(result.summary())
print(result.prediction.label, result.prediction.reasons)
print(result.parameters)                         # {period_days: (value, ±err), ...}
```

## Architecture

```
src/apex_oracle/
  config.py       frozen dataclass configs (config-driven pipeline)
  data.py         LightCurve + loaders: synthetic / CSV / real TESS (lightkurve)
  synthetic.py    light-curve generation + transit injection
  preprocess.py   transit-safe detrending (Savitzky-Golay / wotan) + phase folding
  search.py       BLS-style period search with 2x/3x de-aliasing (TLS optional)
  features.py     robust physical + shape features (depth, duration, SNR, U-vs-V, secondary)
  model.py        classifier registry: rule-based (transparent) + trainable scikit-learn
  evaluate.py     injection-recovery, classification metrics, calibration (ECE)
  pipeline.py     end-to-end orchestration -> InspectionResult
  server.py       zero-dependency JSON API (deploy as a microservice)
  cli.py          apex-oracle {demo, run, train, evaluate, serve}
app/inspector/    browser "mission control" UI (Three.js) — see its README
```

The classifier is swappable via a registry; the **rule-based** model is fully transparent
(physics rules), the **scikit-learn** model is trainable and serialisable. The browser
Inspector (`app/inspector/`) is the front-end for the same logic.

## Validation

`apex-oracle evaluate` reports, on a fresh synthetic test set:
- per-class **precision / recall / F1** and overall accuracy,
- **expected calibration error (ECE)**,
- **injection-recovery completeness vs transit depth** — the credibility core.

The rule-based baseline reaches ~0.9 accuracy on synthetic data; the documented failure modes
are variability-dominated folds and very shallow, low-SNR signals (quantified by completeness).

**Verified on real TESS data.** `python scripts/run_real.py "WASP-18"` downloads 18,299 real
cadences from MAST and recovers the planet correctly: **TRANSIT, P = 0.941 d** (literature
0.9415 d), depth ~11,000 ppm, with the real ~566 ppm secondary eclipse.

**Real labelled benchmark** (`python scripts/benchmark.py` → `outputs/real_benchmark.md`):
on 11 real TESS targets (8 confirmed planets + 3 eclipsing binaries),
**accuracy 0.82, eclipsing-binary recall 1.00 (3/3), planet recall 0.75 (6/8)**.
The misses are *period-search* failures on ultra-hot Jupiters with strong tidal / ellipsoidal
phase-curve modulation (WASP-12, WASP-19) — a well-known hard regime that motivates the TLS /
phase-curve-aware search on the roadmap, not a classifier flaw. Real-data testing is also how
we found and fixed period-alias bugs (the ÷2/÷3/÷4 equal-depth-secondary de-aliasing in
`pipeline.detect`). Covered by `tests/test_real_data.py` (auto-skips offline).

## Known limitations
- Default training/validation is on **synthetic** light curves; not yet validated on labelled
  real TESS data.
- BLS can pick a period alias on sparse / low-SNR data.
- Blend vetting needs pixel-level data (TPF/FFI); the 1-D path exposes a flag placeholder.

## Roadmap
1. Real `lightkurve` ingestion at scale; GP detrend (`celerite2`) + `transitleastsquares`.
2. Train + validate on labelled TOI / confirmed-planet / EB sets; report real metrics.
3. Differentiable `batman` + SBI posteriors; centroid / difference-imaging vetting.
4. Packaging, container, deployment, accessibility, and ISRO security review.

## License
MIT.
