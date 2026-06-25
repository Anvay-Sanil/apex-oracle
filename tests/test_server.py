"""API server tests: start the server in a thread and exercise the endpoints."""
from __future__ import annotations

import json
import threading
import urllib.request


from apex_oracle.server import analyze, create_server


def _post(port: int, path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_analyze_function_synthetic():
    out = analyze({"source": "synthetic:eb"})
    assert out["classification"] in {"transit", "eclipse", "starspot"}
    assert out["parameters"]["period_days"][0] > 0
    assert len(out["folded_curve"]["phase"]) > 10


def test_server_health_and_analyze():
    srv = create_server("127.0.0.1", 0)            # ephemeral port
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            health = json.loads(r.read())
        assert health["status"] == "ok"

        code, out = _post(port, "/analyze", {"source": "synthetic:eb"})
        assert code == 200
        assert out["classification"] == "eclipse"

        code, err = _post(port, "/analyze", {})    # bad request
        assert code == 400
        assert "error" in err
    finally:
        srv.shutdown()
        t.join(timeout=5)
