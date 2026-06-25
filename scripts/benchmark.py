"""Real-TESS labelled benchmark: run APEX-ORACLE on known targets, score it.

Confirmed planets (transit) and eclipsing binaries (eclipse) with catalog periods
serve as ground truth. Targets that fail to download are skipped and reported.

Usage:  python scripts/benchmark.py
Writes: outputs/real_benchmark.md
"""
from __future__ import annotations

from pathlib import Path

from apex_oracle import ExoplanetPipeline
from apex_oracle.data import from_tess

# (target, true_label, catalog_period_days)
TARGETS = [
    ("WASP-18", "transit", 0.94145),
    ("WASP-121", "transit", 1.27493),
    ("WASP-100", "transit", 2.84938),
    ("KELT-9", "transit", 1.48112),
    ("HAT-P-7", "transit", 2.20474),
    ("WASP-19", "transit", 0.78884),
    ("WASP-12", "transit", 1.09142),
    ("AU Mic", "transit", 8.46300),     # young M dwarf, deeper planet b
    ("Algol", "eclipse", 2.86730),      # beta Per, Algol-type EB
    ("RZ Cas", "eclipse", 1.19535),     # Algol-type EB
    ("U Cep", "eclipse", 2.49303),      # Algol-type EB
]


def _period_match(rec: float, cat: float, tol: float = 0.02) -> bool:
    for f in (1.0, 2.0, 0.5):
        if abs(rec / (cat * f) - 1) < tol or abs((rec * f) / cat - 1) < tol:
            return True
    return False


def main() -> int:
    rows, n_ok = [], 0
    for name, label, cat_p in TARGETS:
        try:
            lc = from_tess(name)
            res = ExoplanetPipeline().run(lc)
            pmatch = _period_match(res.period_days, cat_p)
            rows.append((name, label, res.prediction.label,
                         round(res.period_days, 4), cat_p, pmatch,
                         res.prediction.label == label, int(lc.time.size)))
            n_ok += 1
            print(f"OK   {name:10s} true={label:8s} pred={res.prediction.label:8s} "
                  f"P={res.period_days:.4f} (cat {cat_p:.4f}) "
                  f"{'period OK' if pmatch else 'PERIOD OFF'}")
        except Exception as exc:
            rows.append((name, label, "SKIP", None, cat_p, False, False, 0))
            print(f"SKIP {name:10s} ({type(exc).__name__})")

    done = [r for r in rows if r[2] != "SKIP"]
    planets = [r for r in done if r[1] == "transit"]
    ebs = [r for r in done if r[1] == "eclipse"]
    acc = sum(r[6] for r in done) / len(done) if done else 0.0
    p_recall = sum(r[2] == "transit" for r in planets) / len(planets) if planets else 0.0
    eb_recall = sum(r[2] == "eclipse" for r in ebs) / len(ebs) if ebs else 0.0
    p_period = sum(r[5] for r in planets) / len(planets) if planets else 0.0

    print("\n=== REAL BENCHMARK ===")
    print(f"downloaded {len(done)}/{len(TARGETS)} | overall accuracy {acc:.2f}")
    print(f"planet recall {p_recall:.2f} ({len(planets)}) | "
          f"period-match {p_period:.2f} | eclipse recall {eb_recall:.2f} ({len(ebs)})")

    out = Path("outputs"); out.mkdir(exist_ok=True)
    lines = ["# APEX-ORACLE — real TESS benchmark", "",
             f"Downloaded {len(done)}/{len(TARGETS)} targets. "
             f"Overall accuracy **{acc:.2f}**, planet recall **{p_recall:.2f}**, "
             f"period-match **{p_period:.2f}**, eclipse recall **{eb_recall:.2f}**.", "",
             "| target | true | predicted | P_rec (d) | P_cat (d) | period? | correct? | cadences |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | "
                     f"{'yes' if r[5] else 'no'} | {'yes' if r[6] else 'no'} | {r[7]} |")
    (out / "real_benchmark.md").write_text("\n".join(lines) + "\n")
    print("saved outputs/real_benchmark.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
