"""Head-to-head: rule-based vs trained ML classifier on the real TESS benchmark.

Extracts features once per target (via the same pipeline) and classifies with both,
so the comparison is fair. Trains and saves the ML model.
"""
from __future__ import annotations

from pathlib import Path

from apex_oracle.data import from_tess
from apex_oracle.model import RuleBasedClassifier, train_default_model
from apex_oracle.pipeline import detect

TARGETS = [
    ("WASP-18", "transit"), ("WASP-121", "transit"), ("WASP-100", "transit"),
    ("KELT-9", "transit"), ("HAT-P-7", "transit"), ("WASP-19", "transit"),
    ("WASP-12", "transit"), ("AU Mic", "transit"),
    ("Algol", "eclipse"), ("RZ Cas", "eclipse"), ("U Cep", "eclipse"),
]


def main() -> int:
    print("training ML model (gradient-boosted trees on synthetic injections)...")
    ml = train_default_model(n_per_class=100, seed=0)
    Path("models").mkdir(exist_ok=True)
    ml.save("models/apex_oracle.joblib")
    rule = RuleBasedClassifier()

    rows = []
    for name, label in TARGETS:
        try:
            lc = from_tess(name)
            feats = detect(lc)[6]
            rp = rule.predict_one(feats).label
            mp = ml.predict_one(feats).label
            rows.append((label, rp, mp))
            print(f"{name:10s} true={label:8s} rule={rp:8s} ml={mp:8s}"
                  f"  {'' if rp==mp else '  <-- differ'}")
        except Exception as exc:
            print(f"SKIP {name} ({type(exc).__name__})")

    def acc(i):
        return sum(r[i] == r[0] for r in rows) / len(rows) if rows else 0.0

    def recall(i, lab):
        sub = [r for r in rows if r[0] == lab]
        return sum(r[i] == lab for r in sub) / len(sub) if sub else 0.0

    print(f"\nRULE-BASED : acc {acc(1):.2f} | planet-recall {recall(1,'transit'):.2f} "
          f"| eb-recall {recall(1,'eclipse'):.2f}")
    print(f"ML (sklearn): acc {acc(2):.2f} | planet-recall {recall(2,'transit'):.2f} "
          f"| eb-recall {recall(2,'eclipse'):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
