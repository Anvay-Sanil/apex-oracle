"""Append real false-positive (mostly eclipsing-binary) targets from the TOI catalogue.

TOI false positives are guaranteed 2-min SPOC targets with catalogue dispositions -- the
reliable source of real non-planet examples (the by-name classic EBs lack 2-min data).
Appends to data/real_features.csv with label 'eclipse'. Skips failures; flushes per row.
"""
from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

from apex_oracle.data import from_tess
from apex_oracle.features import TransitFeatures
from apex_oracle.pipeline import detect

CSV = Path("data/real_features.csv")
COLS = ["name", "label", *TransitFeatures.vector_names(), "period_days", "n_cadences"]
TARGET_N = 30        # successful FP downloads to add
MAX_ATTEMPTS = 60


def main() -> int:
    done = set()
    if CSV.exists():
        done = {r["name"] for r in csv.DictReader(CSV.open())}

    t = NasaExoplanetArchive.query_criteria(table="toi", select="tid,tfopwg_disp",
                                            where="tfopwg_disp='FP'")
    tids = sorted({int(x) for x in t["tid"]})
    random.Random(1).shuffle(tids)

    fh = CSV.open("a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(COLS); fh.flush()

    got, attempts = 0, 0
    for tic in tids:
        if got >= TARGET_N or attempts >= MAX_ATTEMPTS:
            break
        name = f"TIC {tic}"
        if name in done:
            continue
        attempts += 1
        try:
            lc = from_tess(name)
            feats = detect(lc)[6]
            w.writerow([name, "eclipse", *[round(float(x), 6) for x in feats.to_vector()],
                        round(feats.period_days, 5), int(lc.time.size)])
            fh.flush()
            got += 1
            print(f"OK {got:2d}/{TARGET_N} {name:14s} depth={feats.depth_ppm:.0f} "
                  f"sec/d={feats.secondary_ppm/max(feats.depth_ppm,1):.2f}")
        except Exception as exc:
            print(f"skip {name} ({type(exc).__name__})")
    fh.close()
    print(f"DONE: added {got} false-positive targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
