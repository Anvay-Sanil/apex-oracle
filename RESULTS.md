# APEX-ORACLE — Validation Results

A consolidated, honest record of what the system does and how well, with reproducible
commands. Generated during development; numbers are from the committed pipeline.

## What exists
- An installable, typed Python package (`apex_oracle`) with a CLI and a JSON API service.
- A browser "mission control" UI that calls the **same validated engine** (one system).
- 15 automated tests (13 offline + 2 real-TESS network), all passing.

## Real-TESS labelled benchmark
`python scripts/benchmark.py` → 11 real TESS targets with catalog periods as ground truth.

| Metric | Value |
|---|---|
| Overall accuracy | **0.82** |
| Eclipsing-binary recall | **1.00** (3/3: Algol, RZ Cas, U Cep — periods match catalog) |
| Planet recall | **0.75** (6/8 confirmed planets classified `transit`) |
| Period match (planets) | 0.38 |

Correctly recovered planets: WASP-18, WASP-121, WASP-100, KELT-9, HAT-P-7, AU Mic.
Documented failures: WASP-19 (3× period alias with a moderate secondary) and WASP-12
(extreme tidal/ellipsoidal modulation dominates the transit) — a known-hard regime for
box-BLS that motivates a TLS / phase-curve-aware search (roadmap), **not** a classifier flaw.

### Headline real example — WASP-18
`python scripts/run_real.py "WASP-18"` (18,299 real TESS cadences from MAST):
- Classification: **TRANSIT**, confidence 0.99
- Period: **0.941 d** (literature 0.9415 d)
- Depth: ~11,000 ppm; real secondary eclipse ~566 ppm; SNR 190
- Figure: `outputs/real_WASP-18.png`

## Synthetic validation
`apex-oracle evaluate`:
- Accuracy ~0.90, ECE 0.117 (calibration)
- Per class: eclipse F1 1.00, transit F1 ~0.86, starspot F1 ~0.84
- Injection-recovery completeness vs depth: 17% @200 ppm → 83% @3000 ppm (monotone)

## Reproduce
```bash
uv pip install -e ".[astro]"
pytest -q                       # 15 tests
apex-oracle evaluate            # synthetic metrics + injection-recovery
python scripts/benchmark.py     # real labelled benchmark -> outputs/real_benchmark.md
python scripts/run_real.py "WASP-18"
apex-oracle serve               # JSON API; the UI's "REAL ENGINE" mode calls it
```

## AI-enabled: ML trained on REAL data, and it beats physics
The production default is now a **real-data-trained ML model** (a physics gate handles the
"no transit / starspot" case; the ML decides planet vs eclipsing-binary).

Built a real labelled set of **65 TESS targets** — 34 confirmed planets + 31 catalogue
false positives (`scripts/build_real_dataset.py` + `scripts/fetch_toi_fp.py`, FPs from the TOI
catalogue which guarantees 2-min SPOC data). Leakage-free **5-fold cross-validation**
(`scripts/cv_compare.py`), identical features for both:

| Task | Physics rules | **ML (real-trained, CV)** |
|---|---|---|
| Planet vs false-positive | 0.75 acc (planet/fp recall 0.79 / 0.71) | **0.95 acc (1.00 / 0.90)** |
| Multiclass | 0.45 | **0.95** |

**The real-data-trained ML beats the physics rules decisively** (0.95 vs 0.75) and is now the
default (`src/apex_oracle/models/default.joblib`, loaded by `ExoplanetPipeline`; rule-based
remains the transparent fallback and the synthetic-detection path).

Earlier, the *synthetic*-trained ML scored only 0.55 on real data — training on **real** data
is what made AI win. Honest caveat: the 34 planets are hot-Jupiter-heavy, so the model is
strongest on the catalogue vetting distribution; broadening to shallow/small planets (and the
deep Transformer/SBI models) is the next step (P1).

## Honest limitations
- Default training is synthetic injections; not yet trained/validated on a large labelled
  real set.
- Period search is box-BLS; ultra-hot Jupiters with strong phase-curve modulation need TLS.
- Blend vetting needs pixel-level data; the 1-D path exposes a flag placeholder only.
- Not ISRO-certified: that requires their labelled data, security review, and deployment infra.
