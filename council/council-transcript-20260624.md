# LLM Council Transcript — ISRO Hackathon 2026 Architecture Selection

- Date: 2026-06-24
- Session: Phase 2 — The Triad of Disruptive Architectures
- Chair: Apex Astro-Informatics Architect
- Council: 14 members + Chair

## Framed question

We are building a pipeline for the Bharatiya Antariksh Hackathon 2026 (ISRO):
"AI-enabled detection of exoplanets from noisy astronomical light curves" using TESS
high-cadence data. The pipeline must detect periodic dips, classify them
(transit / eclipse / blend / starspot), estimate physical parameters (period, depth,
duration), report SNR / confidence, and visualize results with explainability. Judges
are physicists who distrust black-box AI. Constraints: ~48h build, scarce trustworthy
labels, single-sector data gaps, crowded-field blending, correlated stellar noise.

Which of three candidate architectures should we build to win 1st place?

- Horizon-A "Velocity": Savitzky-Golay detrend + BLS candidate generation + 1-D TCN
  dual-view classifier. Fast, cheap, deployable. Ceiling capped.
- Horizon-B "Apex": GP detrend (celerite2) + ensemble Time-Series Transformer + 1-D ViT
  + GBT on engineered features, multimodal (flux + centroid + Gaia), attention XAI,
  BNN/MC-dropout confidence. Highest accuracy; heavy; overfit risk on small labels.
- Horizon-C "Oracle": differentiable batman/Mandel-Agol transit model in the loss,
  network predicts physical parameters, joint GP red-noise model, SBI posteriors.
  White-box by construction; recovers monotransits; hardest to ship.

## Advisor independent verdicts

- Vasquez (Disentangler) -> B. Flux-only models cannot beat blends; multimodal
  transformer + attention is the only thing that does, and attention is free XAI.
  Risk: needs enough labels.
- Rao (Estimator) -> C. A differentiable batman makes every latent dimension a physical
  quantity; no judge can call it a black box. Recovers monotransits everyone else
  discards; exact posteriors via SBI.
- Thorne (Detrender) -> not A. SavGol erases ~200 ppm transits; GP detrending is
  mandatory. A is disqualified on science credibility for deep cases. Pick B or C.
- Tanaka (Adjudicator) -> hybrid. 48h + small labeled set means B's heavy ensemble
  overfits. Injection-recovery must be the headline metric. C generalizes better but
  must prove optimization stability.
- Mehta (Conductor / PM) -> hybrid, layered. Pure C will not ship reliably in time;
  pure B is "another big model." Layer it: A in hours, B as spine day 1-2, C only on the
  short candidate list. Bounded scope = de-risked.
- Vikram (Contrarian) -> hybrid + fallback. Judges are physicists who hate black boxes.
  B alone loses the narrative; C alone risks a dead demo. Build the hybrid; if C's
  differentiable model is unstable by hour 30, fall back to classic batman MCMC.
- Krishnan (Storyteller) -> B+C. C's physical parameters are the ultimate explanation;
  B's attention overlays are the ultimate visual. Together they are unbeatable on stage.
- Roth (Inquisitor) -> orthogonal warning. None of the three pure plans foreground
  centroid / difference-imaging vetting — the actual blend killer. Whatever is selected
  MUST bolt this on or we lose to false positives.

## Peer review — blind spots surfaced (anonymized round, mapping revealed)

1. Monotransit recovery (P > ~13.5 d shows only 1-2 transits) was under-weighted by
   everyone except Rao. It is a concrete differentiator.
2. Label scarcity and train/val/test leakage by star/sector is the real 48h constraint;
   most advisors assumed clean labels.
3. Demo latency and confidence calibration: a slow or over-confident model still loses
   on stage. Calibration (ECE, reliability diagrams) must be reported.

## Chairman verdict

### Where the council agrees
Pure Horizon-A does not win 1st. GP detrending is non-negotiable. Injection-recovery,
not accuracy on easy hot Jupiters, is the trust metric. A centroid / difference-imaging
vetting module is mandatory in every option.

### Where the council clashes
Ceiling vs ship-ability. B-camp trusts the ensemble's raw power; C-camp trusts physics
to generalize from scarce labels and to silence the black-box critique. Both are right
about different risks.

### Blind spots caught
Monotransit recovery, label scarcity / leakage, and demo calibration — none fully owned
by any single pure architecture.

### Recommendation
Forge the "APEX-ORACLE" hybrid: Horizon-B multimodal transformer as the detection /
classification spine -> Horizon-C differentiable batman + SBI as the parameter and
confidence head -> Horizon-A BLS as the fast candidate generator -> Roth's vetting module
bolted across all of it. This is the only configuration that maximizes accuracy, kills the
black-box objection, recovers monotransits, and produces a standout demo.

### The one thing to do first
Lock the data + label + injection-recovery harness (Quartermaster + Cartographer +
Adjudicator). Nothing is provable without it.

## Status
Awaiting user selection (Phase 3). Options on the table: pure A, pure B, pure C, or the
recommended APEX-ORACLE hybrid.
