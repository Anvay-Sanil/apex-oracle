# Notes: APEX-ORACLE Pipeline — Decisions & Evidence

## Astrophysical constraints driving the design
- TESS pixel scale ~21 arcsec/px -> crowded-field blending is the #1 false-positive source.
  Mitigation: multimodal model (flux + centroid + Gaia) + difference-imaging vetting.
- Transit depths ~1e2-1e3 ppm sit inside correlated (red) stellar noise. Mitigation: GP
  detrending that preserves transits (wotan/celerite2), never aggressive median/spline.
- ~27.4-day sectors + periodic perigee gaps -> monotransits and period aliasing.
  Mitigation: model-based estimation (differentiable transit) recovers single transits;
  transformer with learned positional encodings tolerates gaps.

## Framework decision
- Primary: PyTorch (repo preference; Transformers/Lightning ecosystem).
- Physics head: torch-native differentiable quadratic-limb-darkened transit (Mandel-Agol /
  Agol 2020). batman-package used as the trusted non-diff validator + MCMC fallback.
- JAX alternative (jaxoplanet + numpyro NUTS) noted as the higher-rigor option if torch
  autodiff through the LD integrals proves unstable. Conductor prefers one stack in 48h.

## Posterior strategy
- Amortized SBI (sbi, neural posterior estimation) trained offline on simulated transits
  -> one-forward-pass posteriors at demo time (fast, impressive).
- dynesty nested sampling for the few headline candidates (gold-standard credible intervals).

## Labels & augmentation
- Positives: confirmed planets + TOI planet candidates (NASA Exoplanet Archive / ExoFOP).
- Negatives/other classes: known eclipsing binaries, false positives, starspot rotators.
- Augmentation: inject batman synthetic transits into real out-of-transit TESS curves to
  balance rare classes and to build the injection-recovery test set.

## Anti-leakage
- GroupKFold by TIC ID AND hold out whole sectors; never let the same star span splits.

## Demo & credibility
- Headline metric = injection-recovery completeness & reliability vs SNR, NOT raw accuracy.
- Report calibration (ECE, reliability diagram) so confidence numbers are trustworthy.
