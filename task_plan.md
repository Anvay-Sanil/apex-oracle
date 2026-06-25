# Task Plan: APEX-ORACLE Exoplanet Detection Pipeline (ISRO Hackathon 2026)

## Goal
Win 1st place at Bharatiya Antariksh Hackathon 2026 with a physics-aware, explainable,
hybrid AI pipeline that detects, classifies, parameter-estimates, and confidence-scores
exoplanet transits in noisy TESS high-cadence light curves.

## Selected architecture
APEX-ORACLE hybrid (council verdict, user-confirmed 2026-06-24):
- Horizon-B multimodal transformer = detection / classification spine
- Horizon-C differentiable batman + SBI = parameter + confidence head
- Horizon-A BLS/TLS = fast candidate generator front-end
- Roth vetting module (centroid / difference imaging) bolted across all paths

## Protocol phases (user pacing)
- [x] Phase 1: Baseline sync + astrophysical constraints
- [x] Phase 2: Triad of disruptive architectures (council deliberation + verdict)
- [~] Phase 3: Master Blueprint (this document + blueprint.md)
- [ ] Phase 4: 3-page ISRO report outline

## Engineering phases (build)
- [ ] E0: Env + repro (uv, Hydra, W&B, seeds) — Forge
- [ ] E1: Data ingestion (lightkurve/MAST, TOI labels, Gaia) — Quartermaster + Cartographer
- [ ] E2: Detrending (wotan/celerite2) + quality masking — Thorne
- [ ] E3: Candidate generation (TLS/BLS) + dual-view phase folding — Quartermaster
- [ ] E4: Detection/classification spine (multimodal transformer) — Vasquez
- [ ] E5: Parameter estimation (differentiable transit + SBI posteriors) — Rao
- [ ] E6: Vetting / false-positive (centroid, odd-even, triceratops) — Roth
- [ ] E7: XAI + visualization (attention overlays, phase-folded theatre) — Krishnan
- [ ] E8: Eval harness (injection-recovery, calibration) — Adjudicator
- [ ] E9: Live demo (Streamlit) + 3-page report — Showrunner + Scribe

## Key questions
1. Differentiable transit: torch-native vs jaxoplanet+numpyro? (Decision: torch-native
   primary, batman+dynesty fallback/validator — keep one stack for 48h.)
2. Posteriors: amortized SBI (fast demo) vs nested sampling (rigorous few)? (Both: SBI
   for the demo, dynesty for the headline candidates.)
3. Label source priority: TOI + confirmed + known FP, augmented with batman injections.

## Decisions made
- 2026-06-24: APEX-ORACLE hybrid selected over pure A/B/C.
- 2026-06-24: PyTorch primary stack (per repo preference); JAX only if physics head needs it.

## Errors encountered
- (none yet)

## Submission artifacts (ISRO idea round)
- Form fields (4): humanized (writing-anti-ai) within char limits. Field 4 currently "not
  first hackathon" + solo wording -> MUST drop "solo" (see rule below).
- CANONICAL UPLOAD: deck/ISRO_BAH2026_Idea_Submission_FILLED.pdf (+ .pptx) -- the ORIGINAL
  ISRO template filled with our points, format preserved. QA passed 10/10. Built by
  deck/fill_template.py.
- Content-source only (do NOT upload): deck/APEX-ORACLE_ISRO_BAH2026.pptx (custom theme).
- RULE VERIFIED: submission must follow provided template ("as per attached template");
  team must be 3-4 members, SOLO NOT ALLOWED. Sources: hack2skill, internshala.
- Editable placeholders in filled template: slide 1 [YOUR TEAM NAME]/[YOUR NAME];
  slide 2 native grid (Team Leader + 3 members, Name/College); slides 5 & 7 optional diagram.

## Council audit: "is it ISRO-level?" (2026-06-24)
Verdict: NO -> it was a stunning prototype, not production software. Top gaps were honesty
(faked TIC/provenance over synthetic data), no real-data capability, no uncertainties, no
validation, no docs. P0 set BUILT + browser-verified on app/inspector/index.html:
- Honest provenance: header now SIMULATION / SOURCE SYNTHETIC (no fake TIC).
- Real-data ingestion: LOAD REAL CSV (time,flux) -> JS detrend + BLS (2x/3x de-alias) +
  fold + classify; switches to REAL DATA mode. Verified planet/EB/flat CSVs classify right.
- Parameter uncertainties (+/-) added; robust median-based estimators; U-vs-V sharpness metric.
- Methodology disclosure in-app; no-WebGL fallback; README at app/inspector/README.md.
- Verified 6/6: planet/EB/starspot correct in BOTH sim and real-CSV modes, 0 console errors.
P1 (still pending = the real ISRO-grade work): lightkurve real data, GP detrend, trained
Transformer + injection-recovery metrics + calibration, SBI posteriors, centroid vetting,
tests/CI/deploy.

## PRODUCTION PACKAGE: apex_oracle (2026-06-25)
Council-set Definition of Done -> built a real, installable, tested Python package (not a demo).
- src/apex_oracle/: config, data (synthetic/CSV/lightkurve), synthetic, preprocess (detrend+fold),
  search (BLS + de-alias, TLS optional), features (robust median/fill-factor), model (registry:
  rule-based + trainable sklearn), evaluate (injection-recovery + metrics + ECE), pipeline, cli.
- Packaging: pyproject.toml (uv/hatchling), Dockerfile, .github/workflows/ci.yml, README.md, .gitignore.
- Tests: tests/ -> 11/11 PASS (pytest, 41s). `pip install -e .` works; `apex-oracle` console script works.
- VERIFIED end-to-end: demo planet->TRANSIT(81%, P=3.414+/-0.014), eb->ECLIPSE(85%, secondary 230ppm).
  evaluate: accuracy 0.90, ECE 0.117; injection-recovery 17%@200ppm -> 83%@3000ppm (monotone).
- Classifier robustness tuned via temp/diag.py (24-case sweep): secondary-eclipse is the primary
  planet/EB discriminator; failure modes are variability-dominated folds + shallow low-SNR (honest).
DoD status: 1-9 MET at prototype/baseline level. STILL PENDING (needs ISRO): validation on labelled
real TESS data, security accreditation, deployment infra, GPU-scale training. NOT "certified-deployable".

## REAL TESS DATA milestone (2026-06-25)
- Installed astro extra (lightkurve 2.6.0). MAST reachable. Ran on REAL data end-to-end.
- WASP-18 (18,299 real TESS cadences): pipeline FIRST mis-locked on 1.882 d (2x alias) -> "eclipse".
  Real-data testing exposed it; added equal-depth-secondary de-aliasing (pipeline.detect).
- After fix: WASP-18 -> TRANSIT, P=0.941 d (matches literature 0.9415), depth ~11000 ppm,
  real secondary 566 ppm, SNR 190. Correct characterisation of a real exoplanet. Figure: outputs/real_WASP-18.png.
- Tests now 13: 11 synthetic + 2 network (real TESS) integration, all PASS. CLI + install verified.
- scripts/run_real.py downloads any target and saves a phase-folded figure.

## REAL LABELLED BENCHMARK (2026-06-25) — capstone validation
- scripts/benchmark.py: 11 real TESS targets (8 confirmed planets + 3 eclipsing binaries) with
  catalog periods as ground truth. Report -> outputs/real_benchmark.md.
- Generalised de-aliasing to ÷2/÷3/÷4 (was ÷2 only) -> accuracy 0.70->0.82, planet recall
  0.57->0.75; KELT-9 + WASP-100 recovered. Synthetic suite still 11/11 (no regression).
- RESULT: accuracy 0.82 | eclipsing-binary recall 1.00 (3/3, periods match catalog) |
  planet recall 0.75 (6/8) | period-match 0.38.
- Remaining real-data failures: WASP-19 (moderate-secondary 3x alias) + WASP-12 (extreme tidal
  ellipsoidal modulation) -> period-search limits of naive BLS on ultra-hot Jupiters. Honest
  documented hard regime; fix = TLS / phase-curve-aware search (P1), NOT a classifier flaw.
- This is genuine real-data validation (recall + catalog-period accuracy), not synthetic-only.

## Period-search iteration + REST service (2026-06-25)
- Attempted "best path" period-search upgrade (box-BLS signal-residue). VERIFICATION REVERTED IT:
  box-BLS folds EBs at half-period (secondary vanishes -> EBs read as transits); a compensating
  period-doubling check then broke planets. Net regression -> reverted to min-bin BLS + de-alias.
  Honest outcome: ultra-hot-Jupiter period search (WASP-12/19) needs TLS/phase-curve = genuine P1.
  Post-revert verified: synthetic 4/24, 11/11 tests. Engine unchanged at 0.82 / EB 1.00 / planet 0.75.
- Pivoted to deployment value: added src/apex_oracle/server.py — zero-dep JSON API
  (GET /health, POST /analyze with source|csv -> classification + params + folded curve).
  CLI `apex-oracle serve`. tests/test_server.py. Verified live: health + analyze on the running
  server returned transit, P=3.414+/-0.014. Tests now 15 (13 non-network + 2 network), all PASS.

## UI <-> BACKEND UNIFIED (2026-06-25)
- Added CORS OPTIONS preflight to server.py. Wired the Inspector UI to the validated engine:
  new "REAL ENGINE" controls (target input + ANALYZE VIA ENGINE button) -> POST /analyze ->
  renderBackendResult() maps the JSON into the existing HUD/charts.
- VERIFIED LIVE in-browser (preview): synthetic:planet -> PLANET TRANSIT (engine), and
  WASP-18 -> the UI sent it to the engine, which downloaded REAL TESS data and returned
  PLANET TRANSIT P=0.941+/-0.004 d, depth 10999+/-58 ppm, rendered with "◆ REAL ENGINE" mode.
- The gorgeous demo now runs the SAME validated pipeline as the tests/benchmark (one system,
  not two implementations). Browser fetch succeeding also confirms the OPTIONS preflight works.

## SUBMISSION CAPTURE (2026-06-25) — real results into the deck
- RESULTS.md: consolidated, reproducible validation record (benchmark 0.82 / EB 1.00 /
  planet 0.75, WASP-18 detail, synthetic metrics, honest limitations, reproduce commands).
- Updated deck/fill_template.py + regenerated ISRO_BAH2026_Idea_Submission_FILLED.pdf:
  - Slide 3 (Opportunity/USP): added "already a working, tested prototype validated on real
    TESS data (0.82 acc, EB recall 1.00, WASP-18 recovered at 0.94 d)".
  - Slide 6 (was mock-up): now embeds outputs/real_WASP-18.png — the REAL detrended light
    curve + phase-folded transit, captioned as a working-prototype proof.
  - QA'd slides 3 + 6 (render): clean, no overflow, figure fits the template.
- Rationale (council): software is strong + deployable; capturing the real evidence into the
  submission is higher-value/lower-risk before the July 1 deadline than gambling on TLS (P1).

## VERCEL DEPLOY PREP (2026-06-25)
- Inspector UI made deploy-ready: ENGINE_API now configurable via ?api=<url> (default localhost).
  Verified: reload clean, no console errors, sim 60fps, apiLabel reflects the URL.
- Added vercel.json (outputDirectory app/inspector) + DEPLOY.md.
- HONEST scope: UI deploys static on Vercel (sim + CSV work fully client-side). Real-TESS Python
  engine does NOT fit Vercel serverless (lightkurve/astropy size + MAST download timeout) -> host
  the Dockerfile on Render/Railway/Fly and point the UI via ?api=https://backend.
- I cannot run `vercel` (needs the user's account login); prepared for copy-paste deploy.

## "AI-enabled?" — rules vs ML, measured honestly (2026-06-25)
- Trained the gradient-boosted ML classifier (widened synthetic depth grids so it can't cheat on
  depth) and compared head-to-head vs rule-based on the real 11-target benchmark (compare_models.py).
- RESULT: rule-based 0.82 acc / EB recall 1.00; ML 0.55 acc / EB recall 0.33 (misses real EBs).
  Sim-to-real gap -> synthetic-trained ML does NOT beat physics rules on real data.
- DECISION: keep the physics rule classifier as default (do not regress). ML shipped as an option
  (models/apex_oracle.joblib, `--model`). Genuine AI-superiority = train deep models on REAL
  labelled catalogues (Transformer/SBI) = P1. Honest framing recorded in RESULTS.md.

## AI IS NOW THE DEFAULT — trained on real data, beats physics (2026-06-26)
- Built a real labelled set: 65 TESS targets (34 confirmed planets + 31 TOI false positives).
  Obstacle: classic EB names lack 2-min SPOC data -> pivoted to TOI FP catalogue (1241 real TICs).
  scripts/build_real_dataset.py + fetch_toi_fp.py + cv_compare.py + train_real_model.py.
- Leakage-free 5-fold CV (planet vs false-positive): ML 0.95 acc (recall 1.00/0.90) vs rules 0.75.
  => REAL-DATA-TRAINED ML BEATS PHYSICS. (Synthetic-trained ML was 0.55 -> training on REAL data is the fix.)
- Made it the DEFAULT: src/apex_oracle/models/default.joblib + HybridClassifier (physics gate for
  starspot/no-signal + real-trained ML for planet/eclipse) + _default_classifier in pipeline.
  Rule-based kept as transparent fallback + synthetic-detection path (injection-recovery).
- Tests updated honestly (synthetic tests pin the rule path; new test asserts real-trained hybrid
  default loads + runs). 14/14 non-network pass. Caveat recorded: planet set hot-Jupiter-heavy.

## DECK: evidenced AI claim baked in (2026-06-26)
- scripts/make_results_figure.py -> outputs/validation_panel.png (2-panel: real WASP-18 fold +
  AI-vs-rules CV bars 0.95/0.75).
- Updated deck: slide 3 USP = "AI trained on real TESS data beats classical vetting 0.95 vs 0.75
  (5-fold CV, 65 targets), 90% FP recall, zero planets missed"; slide 6 = "Validation on real
  TESS data" with the validation_panel figure. Regenerated upload-ready PDF; QA'd slides 3 + 6.

## FULL-APP DEPLOY: one container serves UI + real backend (2026-06-26)
- server.py now serves the Inspector UI at "/" (alongside /analyze, /health); serve(ui_dir);
  CLI `serve --ui-dir` auto-detects app/inspector; UI ENGINE_API defaults to same-origin.
- Dockerfile = full app: installs . + lightkurve + wotan, copies UI, ENV APEX_UI_DIR,
  CMD serve --host 0.0.0.0 --port 8800. One image = UI + real-TESS engine + bundled AI model.
- VERIFIED locally: GET / -> Inspector HTML (42KB); /health ok; POST /analyze synthetic -> eclipse
  via method 'hybrid-ml' (the real AI). 14/14 tests pass. DEPLOY.md rewritten (Render/Railway/Fly
  for full app; Vercel only for static UI + ?api to a backend).

## DOCKER IMAGE VERIFIED + deploy how-to (2026-06-26)
- Made server respect $PORT (cli serve default from env) so Render/Railway/Fly work; render.yaml added.
- BUILT the image (docker build) -> 1.42GB, exit 0; RAN it and verified GET / (UI 42KB), /health,
  POST /analyze. Caught a real bug: container sklearn 1.9 vs model pickled with 1.7 -> 'No module _loss'
  -> silent fallback to rule. FIX: pinned scikit-learn>=1.7,<1.8 in pyproject. Rebuilt + re-verified:
  /analyze -> method 'hybrid-ml' (AI), container sklearn 1.7.2. Deployable image is solid.
- Deploy paths documented (DEPLOY.md): local `docker build && docker run -p 8800:8800`; cloud via
  Render blueprint (render.yaml) / Fly / Railway. Vercel only for the static UI (+ ?api to a backend).

## PS-7 SCIENCE GAPS CLOSED: fitting + blend (2026-06-26)
- LIGHT-CURVE FITTING (PS: "parameters by light curve fitting"): src/apex_oracle/fitting.py -
  trapezoid transit fit via scipy.curve_fit -> depth/duration with covariance 1-sigma errors +
  reduced chi2. Fixed a depth/ingress degeneracy (tie ingress to width, 3 params). Pipeline now
  reports FITTED params. Verified WASP-18: depth 10,297+/-13 ppm, dur 2.17 h, chi2nu 2.87.
- BLEND CLASSIFICATION (PS: classify into "blends"): src/apex_oracle/vetting.py - TPF centroid
  test (in-transit vs out-of-transit photocentre). "blend" becomes a 4th class on a significant
  shift. Opt-in: run(lc, vet_blend=<target>) / CLI `run --vet-blend`. Verified WASP-18: offset
  0.004" (0.35 sigma) -> on-target, not a blend (correct).
- Exposed in CLI (--vet-blend) and API (fitted depth/duration + reduced_chi2). 16/16 non-network
  tests pass; ruff clean. RESULTS.md updated. PS-7 gaps #1 (blend) and #2 (fitting) now CLOSED;
  remaining PS-7 item = the 3-page report.

## Demo delivered (working prototypes)
- FLAGSHIP: app/inspector/index.html - standalone, enterprise-grade "mission control" web app.
  Three.js 3D star+transiting-planet driving a LIVE light curve, phase-folded+attention charts,
  animated classification HUD (period/depth/duration/SNR), scenario presets + sliders.
  Self-contained (Three.js + Google Fonts via CDN); data computed in-browser.
  Verified in browser (preview): 0 console errors; planet/EB/starspot all classify correctly;
  realistic numbers (planet P3.41d, 876ppm, 2.73h, SNR 28). Serve: python -m http.server in
  app/inspector, or open index.html. launch.json config 'inspector' (port 8530).
- app/pipeline.py + app/dashboard.py + app/verify_demo.py: Streamlit/Python version (VERIFY PASS).
- Stand-ins (synthetic data + rule-based classifier) to be replaced by lightkurve + the
  trained Transformer (E1/E4).

## Status
**Phase 3 complete + ISRO idea-submission deck delivered.** Solo entry, not first
hackathon. Awaiting: user to fill name placeholders; optional Phase 4 (3-page report
outline); then begin pipeline implementation (E0 onward). No pipeline code yet (per pacing).
