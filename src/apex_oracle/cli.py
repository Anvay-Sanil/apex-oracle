"""Command-line interface: apex-oracle {demo,run,train,evaluate}."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import PipelineConfig
from .data import from_synthetic, load_lightcurve
from .pipeline import ExoplanetPipeline
from .utils import get_logger, set_seed

logger = get_logger("apex_oracle.cli")


def _print_result(res) -> None:
    f = res.features
    print("\n" + "=" * 64)
    print(res.summary())
    print("-" * 64)
    print(f"classification : {res.prediction.label.upper()}  "
          f"({res.prediction.confidence:.0%}, {res.prediction.method})")
    for r in res.prediction.reasons:
        print(f"   - {r}")
    print(f"vetting        : {json.dumps(res.vetting, default=str)}")
    print("=" * 64 + "\n")


def cmd_demo(args) -> int:
    set_seed(args.seed)
    lc = from_synthetic(kind=args.kind, seed=args.seed)
    _print_result(ExoplanetPipeline(PipelineConfig(seed=args.seed)).run(lc))
    return 0


def cmd_run(args) -> int:
    set_seed(args.seed)
    lc = load_lightcurve(args.source)
    classifier = None
    if args.model:
        from .model import SklearnClassifier
        classifier = SklearnClassifier.load(args.model)
    _print_result(ExoplanetPipeline(PipelineConfig(seed=args.seed), classifier).run(lc))
    return 0


def cmd_train(args) -> int:
    set_seed(args.seed)
    from .model import train_default_model
    logger.info("training default model (%d examples/class)...", args.n)
    clf = train_default_model(n_per_class=args.n, seed=args.seed)
    clf.save(args.out)
    logger.info("saved trained model -> %s", args.out)
    return 0


def cmd_evaluate(args) -> int:
    set_seed(args.seed)
    from .evaluate import evaluate_classifier, injection_recovery
    classifier = None
    if args.model:
        from .model import SklearnClassifier
        classifier = SklearnClassifier.load(args.model)
    rep = evaluate_classifier(classifier, n_per_class=args.n, seed=args.seed)
    inj = injection_recovery(seed=args.seed)
    print("\n=== CLASSIFICATION (n=%d) ===" % rep.n)
    print(f"accuracy {rep.accuracy:.3f} | ECE {rep.ece:.3f}")
    for label, m in rep.per_class.items():
        print(f"  {label:9s} P={m['precision']:.2f} R={m['recall']:.2f} "
              f"F1={m['f1']:.2f} (n={m['support']})")
    print("=== INJECTION-RECOVERY (completeness vs depth ppm) ===")
    for depth, comp in inj.items():
        bar = "#" * int(comp * 20)
        print(f"  {depth:6d} ppm | {comp:4.0%} {bar}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="apex-oracle", description="APEX-ORACLE exoplanet pipeline")
    p.add_argument("--seed", type=int, default=42)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run on a synthetic signal")
    d.add_argument("--kind", choices=["planet", "eb", "starspot"], default="planet")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="run on a source (CSV path, TIC/TOI id, or synthetic:<kind>)")
    r.add_argument("source")
    r.add_argument("--model", help="path to a trained .joblib model (optional)")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("train", help="train the default classifier on synthetic injections")
    t.add_argument("--out", default="apex_oracle_model.joblib")
    t.add_argument("--n", type=int, default=60, help="examples per class")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="classification metrics + injection-recovery")
    e.add_argument("--model", help="path to a trained .joblib model (optional)")
    e.add_argument("--n", type=int, default=40, help="test examples per class")
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("serve", help="run the JSON API server (and the UI if found)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8800")))
    s.add_argument("--ui-dir", default=None, help="serve the Inspector UI from this dir at /")
    s.set_defaults(func=cmd_serve)
    return p


def cmd_serve(args) -> int:
    import os
    from pathlib import Path
    from .server import serve
    ui = args.ui_dir or os.environ.get("APEX_UI_DIR")
    if ui is None:  # auto-detect the repo UI
        for cand in ("app/inspector", str(Path(__file__).resolve().parents[2] / "app" / "inspector")):
            if (Path(cand) / "index.html").exists():
                ui = cand
                break
    serve(args.host, args.port, ui)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
