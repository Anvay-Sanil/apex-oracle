"""Unified light-curve loading: synthetic, CSV, or real TESS via lightkurve."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .synthetic import SyntheticConfig, generate_lightcurve


@dataclass(frozen=True)
class LightCurve:
    time: np.ndarray
    flux: np.ndarray
    source: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time.shape != self.flux.shape:
            raise ValueError("time and flux must have the same shape")
        if self.time.size < 10:
            raise ValueError("light curve too short (need >= 10 points)")


def from_synthetic(**kwargs) -> LightCurve:
    cfg = SyntheticConfig(**kwargs)
    t, f = generate_lightcurve(cfg)
    return LightCurve(time=t, flux=f, source=f"synthetic:{cfg.kind}", meta={"config": cfg})


def from_csv(path: str | Path) -> LightCurve:
    """Load a two-column (time, flux) CSV/TXT; headers and '#' comments are skipped."""
    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s[0] in "#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            continue
        parts = s.replace(",", " ").replace(";", " ").split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    if len(rows) < 10:
        raise ValueError(f"{path}: fewer than 10 valid (time, flux) rows parsed")
    arr = np.asarray(rows, dtype=float)
    return LightCurve(time=arr[:, 0], flux=arr[:, 1], source=f"csv:{path.name}")


def from_tess(target: str, **kwargs) -> LightCurve:  # pragma: no cover - needs network + lightkurve
    """Download a real TESS light curve via lightkurve (requires the `astro` extra)."""
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise ImportError(
            "Real TESS ingestion needs the optional 'astro' extra: "
            "pip install 'apex-oracle[astro]'"
        ) from exc
    sr = lk.search_lightcurve(target, mission="TESS", author="SPOC", **kwargs)
    if len(sr) == 0:
        raise ValueError(f"No SPOC light curve found for {target}")
    lc = sr.download().remove_nans().normalize()
    return LightCurve(
        time=np.asarray(lc.time.value, dtype=float),
        flux=np.asarray(lc.flux.value, dtype=float),
        source=f"tess:{target}",
        meta={"mission": "TESS"},
    )


def load_lightcurve(source: str, **kwargs) -> LightCurve:
    """Dispatch by source: a .csv/.txt path, 'TIC ...' / 'TOI ...', or 'synthetic:<kind>'."""
    if source.startswith("synthetic"):
        kind = source.split(":", 1)[1] if ":" in source else "planet"
        return from_synthetic(kind=kind, **kwargs)
    p = Path(source)
    if p.suffix.lower() in {".csv", ".txt", ".dat"} and p.exists():
        return from_csv(p)
    return from_tess(source, **kwargs)
