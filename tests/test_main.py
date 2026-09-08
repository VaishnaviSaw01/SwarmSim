"""Integration tests for main.py -- the FastAPI layer around the pipeline.

Uses FastAPI's TestClient (no real network / server process needed).
Each test that hits POST /simulate uses small agent/round counts to stay
fast and to leave headroom under the module-level rate limiter, which is
shared across every test in this file since it lives on the `main`
module for the process's lifetime.
"""

import main
from ratelimit import RateLimiter

try:
    from starlette.testclient import TestClient
except ImportError:  # pragma: no cover
    from fastapi.testclient import TestClient

client = TestClient(main.app)


def test_index_page_serves_the_demo_form():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<form" in resp.text
    assert "SwarmSim" in resp.text


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_simulate_returns_report_and_chart_is_fetchable():
    resp = client.post(
        "/simulate",
        json={"topic": "remote work policy", "num_agents": 8, "num_rounds": 5},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert set(data) >= {"run_id", "metrics", "chart_url", "elapsed_seconds"}
    assert data["chart_url"] == f"/chart/{data['run_id']}"
    assert "summary" in data["metrics"]

    chart_resp = client.get(data["chart_url"])
    assert chart_resp.status_code == 200
    assert chart_resp.headers["content-type"] == "image/png"


def test_simulate_rejects_params_over_the_public_cap():
    resp = client.post(
        "/simulate",
        json={"topic": "x", "num_agents": 5000, "num_rounds": 5},
    )
    assert resp.status_code == 422  # Pydantic validation, caught before any work runs


def test_chart_endpoint_rejects_non_numeric_run_id():
    # Guards against path-traversal-shaped run_ids ever reaching the filesystem.
    resp = client.get("/chart/../../etc/passwd")
    assert resp.status_code in (400, 404)  # 404 if the route itself doesn't match


def test_chart_endpoint_404s_for_a_well_formed_but_unknown_run_id():
    resp = client.get("/chart/999999999999999")
    assert resp.status_code == 404


def test_rate_limit_returns_429_once_exceeded(monkeypatch):
    monkeypatch.setattr(main, "_simulate_limiter", RateLimiter(max_requests=1, window_seconds=60))

    first = client.post("/simulate", json={"topic": "x", "num_agents": 4, "num_rounds": 2})
    second = client.post("/simulate", json={"topic": "x", "num_agents": 4, "num_rounds": 2})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Retry-After" in second.headers
