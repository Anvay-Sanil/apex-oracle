"""Two-panel validation figure for the deck: real WASP-18 detection + CV (AI vs rules)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from apex_oracle import ExoplanetPipeline
from apex_oracle.data import from_tess
from apex_oracle.preprocess import binned_profile

# Panel A: real WASP-18 phase-folded transit
res = ExoplanetPipeline().run(from_tess("WASP-18"))
phase, folded = res.arrays["phase"], res.arrays["folded"]
centers, prof = binned_profile(phase, folded, 240)

# Panel B: leakage-free 5-fold CV on 65 real targets (planet vs false-positive)
metrics = ["Accuracy", "Planet\nrecall", "False-pos\nrecall"]
rules = [0.75, 0.79, 0.71]
ml = [0.95, 1.00, 0.90]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.7))
ax[0].plot(phase, folded, ".", ms=1.2, color="#9aa0aa", alpha=0.4)
ax[0].plot(centers, prof, "-", color="#0F6E56", lw=2.2)
ax[0].set_xlim(-0.12, 0.12)
ax[0].set_title("Real TESS detection — WASP-18 (P = 0.94 d)", fontsize=11)
ax[0].set_xlabel("phase"); ax[0].set_ylabel("normalised flux")

x = np.arange(3); w = 0.36
ax[1].bar(x - w / 2, rules, w, label="Physics rules", color="#B6BCC8")
ax[1].bar(x + w / 2, ml, w, label="AI (real-trained)", color="#185FA5")
ax[1].set_xticks(x); ax[1].set_xticklabels(metrics, fontsize=9)
ax[1].set_ylim(0, 1.08)
ax[1].set_title("Planet vs false-positive — 5-fold CV, 65 real targets", fontsize=11)
ax[1].legend(loc="lower right", fontsize=9, framealpha=0.9)
for i, (r, m) in enumerate(zip(rules, ml)):
    ax[1].text(i - w / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8, color="#555")
    ax[1].text(i + w / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8, fontweight="bold", color="#0c447c")
fig.tight_layout()
fig.savefig("outputs/validation_panel.png", dpi=130)
print("saved outputs/validation_panel.png")
