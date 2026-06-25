"""Build a REAL labelled feature set from catalogue TESS targets.

Curated names with known labels (confirmed planets = transit, known eclipsing binaries =
eclipse, active/rotational variables = starspot). For each: download (cached) -> run the
pipeline -> extract features -> append a row to data/real_features.csv.

Robust: any target that fails to download/process is skipped and logged. Rows are flushed
incrementally so a hang never loses prior progress. Re-running skips targets already in the CSV.
"""
from __future__ import annotations

import csv
from pathlib import Path

from apex_oracle.data import from_tess
from apex_oracle.features import TransitFeatures
from apex_oracle.pipeline import detect

# (name, label) -- generously over-provisioned; failures are skipped.
PLANETS = [
    "WASP-18", "WASP-19", "WASP-12", "WASP-121", "WASP-100", "WASP-4", "WASP-5", "WASP-46",
    "WASP-77 A", "WASP-95", "WASP-126", "WASP-62", "WASP-43", "WASP-64", "WASP-65", "WASP-72",
    "WASP-78", "WASP-97", "WASP-110", "WASP-120", "KELT-9", "KELT-7", "KELT-11", "KELT-16",
    "KELT-20", "KELT-24", "KELT-1", "HAT-P-7", "TrES-2", "HD 209458", "HD 189733", "WASP-33",
    "AU Mic", "WASP-101", "WASP-103",
]
ECLIPSES = [
    "Algol", "RZ Cas", "U Cep", "AR Lac", "RT And", "RW Tau", "SZ Her", "TX UMa", "U Sge",
    "AR Aur", "TV Cas", "TW Dra", "RW Gem", "AB And", "SW Lac", "WW Cyg", "V505 Sgr", "DM Per",
    "EE Peg", "CV Boo", "AI Hya", "RS CVn", "Z Her", "V1143 Cyg", "DI Her",
]
STARSPOTS = ["AB Dor", "BY Dra", "V410 Tau", "LO Peg", "EK Dra", "PZ Tel", "V374 Peg", "AP Col"]

TARGETS = ([(n, "transit") for n in PLANETS]
           + [(n, "eclipse") for n in ECLIPSES]
           + [(n, "starspot") for n in STARSPOTS])

CSV = Path("data/real_features.csv")
COLS = ["name", "label", *TransitFeatures.vector_names(), "period_days", "n_cadences"]


def main() -> int:
    CSV.parent.mkdir(exist_ok=True)
    done = set()
    if CSV.exists():
        with CSV.open() as fh:
            done = {row["name"] for row in csv.DictReader(fh)}
    new = CSV.exists()
    fh = CSV.open("a", newline="")
    writer = csv.writer(fh)
    if not new:
        writer.writerow(COLS)
        fh.flush()

    ok = len(done)
    for i, (name, label) in enumerate(TARGETS, 1):
        if name in done:
            continue
        try:
            lc = from_tess(name)
            feats = detect(lc)[6]
            writer.writerow([name, label, *[round(float(x), 6) for x in feats.to_vector()],
                             round(feats.period_days, 5), int(lc.time.size)])
            fh.flush()
            ok += 1
            print(f"[{i}/{len(TARGETS)}] OK   {name:12s} {label:8s} "
                  f"depth={feats.depth_ppm:.0f} sec/d={feats.secondary_ppm/max(feats.depth_ppm,1):.2f}")
        except Exception as exc:
            print(f"[{i}/{len(TARGETS)}] SKIP {name:12s} ({type(exc).__name__})")
    fh.close()
    print(f"\nDONE: {ok} labelled targets -> {CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
