# Master Blueprint — APEX-ORACLE Exoplanet Detection Pipeline

ISRO Bharatiya Antariksh Hackathon 2026 · TESS noisy light curves
Architecture: APEX-ORACLE hybrid · Stack: Python 3.11 + PyTorch · Last updated 2026-06-24

---

## 0. End-to-end data flow

```
TESS data (MAST)
  -> [E1] Ingestion        (lightkurve, astroquery)        -> raw LightCurve + metadata
  -> [E2] Detrending       (wotan, celerite2, astropy)     -> flattened flux (transit-safe)
  -> [E3] Candidate + Fold (transitleastsquares, lightkurve)-> period/t0/dur + dual views
  -> [E4] Neural inference (PyTorch transformer, XGBoost)  -> class + prob + attention
  -> [E5] Param estimation (diff. transit, gpytorch, sbi)  -> physical params + posteriors
  -> [E6] Vetting          (lightkurve centroids, triceratops)-> FP flags + FPP + disposition
  -> [E7] XAI + viz        (matplotlib, plotly)            -> attention-overlaid figures
  -> [E8] Eval harness     (batman injection, sklearn)     -> completeness/reliability/ECE
  -> [E9] Demo + report    (streamlit, plotly)             -> live app + 3-page PDF
```

Owners (council): E0 Forge · E1 Quartermaster+Cartographer · E2 Thorne · E3 Quartermaster
· E4 Vasquez · E5 Rao · E6 Roth · E7 Krishnan · E8 Adjudicator · E9 Showrunner+Scribe.

---

## 1. Exact libraries (install set)

Core ML:        torch, pytorch-lightning, timm, transformers, xgboost, scikit-learn
Astro/data:     lightkurve, astropy, astroquery, transitleastsquares, wotan
Detrend/GP:     wotan, celerite2, gpytorch
Physics/Bayes:  batman-package, sbi, dynesty, emcee ; (optional JAX: jaxoplanet, numpyro)
Vetting:        triceratops, lightkurve (centroids/difference imaging)
Config/track:   hydra-core, omegaconf, wandb
Viz/app:        matplotlib, plotly, streamlit
Utils:          numpy, pandas, scipy, h5py, pyarrow
Env:            uv (manager), python 3.11

`uv` bootstrap:
```
uv init exoplanet-apex-oracle && cd exoplanet-apex-oracle
uv add torch pytorch-lightning timm transformers xgboost scikit-learn \
       lightkurve astropy astroquery transitleastsquares wotan \
       celerite2 gpytorch batman-package sbi dynesty emcee triceratops \
       hydra-core omegaconf wandb matplotlib plotly streamlit \
       numpy pandas scipy h5py pyarrow
```

---

## 2. Directory structure (coding-style compliant, 200-400 line files)

```
exoplanet-apex-oracle/
|- conf/                         # Hydra configs
|  |- config.yaml
|  |- data/ detrend/ fold/ model/ estimate/ vet/ eval/   # per-stage groups
|- src/
|  |- data_module/               # E1: ingestion + labels + Gaia
|  |  |- __init__.py             # DatasetFactory + register_dataset
|  |  |- ingest.py  labels.py  gaia.py  cache.py
|  |- preprocess_module/         # E2 + E3
|  |  |- detrend.py  mask.py  candidate.py  fold.py
|  |- model_module/              # E4: spine (factory/registry)
|  |  |- __init__.py             # ModelFactory + register_model
|  |  |- transformer_branch.py  tabular_branch.py  fusion.py  heads.py
|  |- physics_module/            # E5: ORACLE
|  |  |- transit_torch.py        # differentiable Mandel-Agol (quadratic LD)
|  |  |- gp_noise.py  estimate.py  sbi_posterior.py  batman_fallback.py
|  |- vetting_module/            # E6
|  |  |- odd_even.py  secondary.py  centroid.py  fpp.py
|  |- xai_module/                # E7
|  |  |- attention.py  figures.py
|  |- utils/                     # seed, logging, env, metrics
|- run/
|  |- pipeline/ end_to_end.py  train.py  evaluate.py
|  |- outputs/                   # Hydra timestamped runs
|- app/  dashboard.py            # E9 Streamlit demo
|- report/  isro_report.md       # E9 3-page report
|- tests/
```

Patterns: config-driven models (`__init__(self, cfg)`), `@register_model`, `@dataclass(frozen=True)`
configs, type hints, module-level loggers, `__all__` in every `__init__.py`.

---

## 3. Stage specs

### E0 — Environment & reproducibility (Forge)
- uv-managed venv, Python 3.11; Hydra config tree; W&B run logging.
- `set_seed(42)` across random/numpy/torch/cuda; cudnn deterministic.
- Hydra auto-saves resolved config + overrides per run under run/outputs/.

### E1 — Data ingestion (Quartermaster + Cartographer)
- `lightkurve.search_lightcurve(target, mission='TESS', author='SPOC', exptime=120)` for
  2-min PDCSAP; `exptime=20` for 20-s where available; `search_targetpixelfile` for TPFs;
  `search_tesscut` for FFI cutouts (centroid/difference imaging input).
- Labels via NASA Exoplanet Archive / ExoFOP (astroquery.ipac.nexsci): confirmed planets,
  TOI candidates, known eclipsing binaries, false positives. Taxonomy: transit / eclipse /
  blend / starspot.
- Gaia DR3 neighbours via astroquery.gaia within the aperture -> contamination ratio.
- TIC stellar params via astroquery.mast.Catalogs (Teff, R*, logg).
- Cache standardized arrays to parquet/HDF5. Output: (flux, time, quality, meta, label).

### E2 — Detrending (Thorne)
- Quality mask using TESS quality flags; drop NaNs; astropy sigma_clip outliers.
- Median-normalize. Detrend with `wotan.flatten(..., method='biweight'|'gp', window_length=..)`
  tuned to preserve transit timescales; `celerite2` RotationTerm + Matern32Term for the hard
  active stars (joint stellar-activity model). Stitch multi-sector via lightkurve.
- Guardrail: validate detrend does NOT suppress injected synthetic transits (Thorne's veto).

### E3 — Candidate generation + phase folding (Quartermaster, Horizon-A role)
- `transitleastsquares` (TLS) primary period search (realistic transit template -> higher SDE,
  returns period, T0, depth, duration, SDE, odd/even, secondary). astropy BoxLeastSquares as
  a fast cross-check. Monotransit branch: matched-filter single-transit search.
- Phase fold (lightkurve `fold`) -> global view (e.g. 2001 bins over full phase) + local view
  (201 bins zoomed on transit), AstroNet dual-view tensors per candidate.

### E4 — Detection / classification spine (Vasquez, Horizon-B)
- Branch 1 (sequence): Time-Series Transformer / 1-D ViT over global+local views; learned
  positional encodings absorb gaps; CLS token embedding.
- Branch 2 (tabular): TLS features + odd-even depth diff + secondary depth + duration + SNR +
  Gaia contamination + centroid offset + stellar params -> XGBoost / MLP embedding.
- Fusion: concat embeddings -> 4-class head (transit / eclipse / blend / starspot).
- Confidence: MC-dropout or deep ensemble -> calibrated class probabilities.
- XAI hook: store attention weights for E7.

### E5 — Parameter estimation (Rao, Horizon-C ORACLE)
- `physics_module/transit_torch.py`: differentiable quadratic-LD Mandel-Agol transit as a
  torch.nn.Module (params: P, t0, Rp/R*, a/R*, inc, u1, u2, e, omega).
- Joint fit: minimize reconstruction loss (model vs detrended flux) + GP red-noise likelihood
  (`gpytorch`) + physics priors, via gradient descent. batman-package validates the forward
  model and serves as the emcee/dynesty MCMC fallback.
- Posteriors: amortized SBI (`sbi` NPE) trained on simulated transits -> instant credible
  intervals at demo; `dynesty` nested sampling for headline candidates.
- Outputs: period, depth (Rp/R*)^2, duration, a/R*, inc, SNR + full posterior + 1-sigma bounds.

### E6 — Vetting / false-positive (Roth)
- Odd-even depth consistency; secondary-eclipse search at phase 0.5 (EB signature).
- Centroid / difference imaging on TPF/FFI (lightkurve centroids) -> in-transit centroid shift
  flags a blended/off-target source (the #1 FP killer).
- Stellar-density consistency: a/R* from fit vs density from Gaia/TIC.
- `triceratops` statistical false-positive probability using Gaia neighbours.
- Output: vetting flags + FPP + disposition (planet candidate / false positive).

### E7 — XAI + visualization (Krishnan, Pillar 4)
- Phase-folded curve overlaid with fitted transit model + attention heatmap (which points
  drove the class). Attention rollout for the transformer.
- Companion plots: TLS periodogram, odd-even, secondary, centroid, posterior corner.
- matplotlib (publication figures) + plotly (interactive).

### E8 — Evaluation harness (Adjudicator)
- Injection-recovery: inject batman transits into real out-of-transit TESS flux across a
  depth-period-SNR grid; measure completeness and reliability.
- Metrics: precision/recall, PR-AUC (rare positives), completeness vs SNR, reliability
  (1 - FPR), calibration (ECE + reliability diagram).
- Splits: GroupKFold by TIC ID + whole-sector holdout (no leakage).

### E9 — Live demo + report (Showrunner + Scribe)
- Streamlit dashboard: input TIC ID / upload light curve -> run pipeline -> show class,
  params + posteriors, attention overlay, vetting verdict in real time.
- 3-page ISRO report (see Phase 4 outline) with methodology, results, figures.

---

## 4. Build order (48h-aware, Conductor)
E0 -> E1 -> E2/E3 (data spine) -> E8 harness early (so everything is measurable) ->
E4 spine -> E6 vetting -> E5 ORACLE head -> E7 XAI -> E9 demo+report.
Rationale: lock measurability (E8) before models; ship a working A-grade path before
attempting the C-grade physics head; keep batman+dynesty as the always-works fallback.
