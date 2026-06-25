"""Zero-dependency HTTP JSON API for the APEX-ORACLE pipeline.

A light wrapper so the validated engine can be deployed as a service and called by
a UI or batch jobs.  Endpoints:

    GET  /health            -> {"status": "ok", "version": ...}
    POST /analyze           -> run the pipeline and return a JSON result
        body: {"source": "synthetic:planet"} | {"source": "TIC 307210830"}
              | {"csv": "<time,flux text>"}

Run with:  apex-oracle serve --host 127.0.0.1 --port 8800
"""
from __future__ import annotations

import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__
from .config import PipelineConfig
from .data import from_csv, load_lightcurve
from .pipeline import ExoplanetPipeline, InspectionResult
from .utils import get_logger

logger = get_logger("apex_oracle.server")

_UI_DIR: str | None = None   # when set, the Inspector UI is served at "/"


def result_to_dict(res: InspectionResult, n_curve: int = 200) -> dict:
    """Serialise an InspectionResult to a JSON-friendly dict (with a folded curve)."""
    f = res.features
    phase, folded = res.arrays["phase"], res.arrays["folded"]
    step = max(1, phase.size // n_curve)
    return {
        "source": res.source,
        "classification": res.prediction.label,
        "confidence": round(res.prediction.confidence, 4),
        "method": res.prediction.method,
        "reasons": res.prediction.reasons,
        "parameters": {
            "period_days": [round(f.period_days, 5), round(f.period_err_days, 5)],
            "depth_ppm": [round(f.depth_ppm, 1), round(f.depth_err_ppm, 1)],
            "duration_hours": [round(f.duration_hours, 3), round(f.duration_err_hours, 3)],
            "snr": round(f.snr, 2),
        },
        "vetting": res.vetting,
        "folded_curve": {
            "phase": [round(float(x), 5) for x in phase[::step]],
            "flux": [round(float(x), 7) for x in folded[::step]],
        },
    }


def analyze(payload: dict, config: PipelineConfig | None = None) -> dict:
    """Core handler: build a light curve from the payload and run the pipeline."""
    if "csv" in payload:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write(payload["csv"])
            tmp = fh.name
        try:
            lc = from_csv(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
    elif "source" in payload:
        lc = load_lightcurve(payload["source"])
    else:
        raise ValueError("payload must contain 'source' or 'csv'")
    return result_to_dict(ExoplanetPipeline(config or PipelineConfig()).run(lc))


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # quiet default logging
        return

    def do_OPTIONS(self) -> None:           # CORS preflight for browser fetch
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_file(self, path: Path, ctype: str) -> None:
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self._send(404, {"error": "ui not found"})

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            self._send(200, {"status": "ok", "version": __version__})
        elif path in ("", "/index.html") and _UI_DIR is not None:
            self._send_file(Path(_UI_DIR) / "index.html", "text/html; charset=utf-8")
        elif path == "":
            self._send(200, {"status": "ok", "version": __version__, "ui": "not bundled"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/analyze":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, analyze(payload))
        except Exception as exc:  # report errors as JSON, never crash the server
            logger.warning("analyze failed: %s", exc)
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})


def create_server(host: str = "127.0.0.1", port: int = 8800) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _Handler)


def serve(host: str = "127.0.0.1", port: int = 8800, ui_dir: str | None = None) -> None:  # pragma: no cover
    global _UI_DIR
    _UI_DIR = ui_dir
    srv = create_server(host, port)
    extra = " (+ UI at /)" if ui_dir and (Path(ui_dir) / "index.html").exists() else ""
    logger.info("APEX-ORACLE serving on http://%s:%d%s", host, port, extra)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
