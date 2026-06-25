"""APEX-ORACLE: explainable exoplanet transit detection for noisy TESS light curves.

Public API:
    from apex_oracle import ExoplanetPipeline, LightCurve, load_lightcurve
"""
from __future__ import annotations

from .config import PipelineConfig
from .data import LightCurve, load_lightcurve
from .pipeline import ExoplanetPipeline, InspectionResult

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "LightCurve",
    "load_lightcurve",
    "ExoplanetPipeline",
    "InspectionResult",
    "__version__",
]
