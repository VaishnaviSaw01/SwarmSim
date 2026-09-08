"""
main.py -- FastAPI wrapper around the simulate -> evaluate -> report
pipeline.

WHY a single POST endpoint: this project's interesting surface area is the
simulation pipeline itself (agent.py through report.py), all of which is
plain, synchronous Python you can run and test without a server at all
(see simulate.py and tests/). The FastAPI layer exists only to demonstrate
that the pipeline is easy to wrap as a service -- a real "environment
engineering" project needs a way for something else (an eval harness, a
UI, a script) to drive it over HTTP.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from evaluate import evaluate
from report import generate_chart, generate_report
from simulate import run_simulation

app = FastAPI(title="SwarmSim", description="Multi-agent opinion simulation engine")

OUTPUT_DIR = Path(__file__).parent / "output"


class SimulateRequest(BaseModel):
    """Request body for POST /simulate."""

    topic: str = Field(..., min_length=1, description="Topic text to seed opinions from")
    num_agents: int = Field(30, ge=4, le=500, description="Number of agents in the swarm")
    num_rounds: int = Field(30, ge=1, le=500, description="Number of simulation rounds")


@app.post("/simulate")
def simulate(request: SimulateRequest) -> dict:
    """Run one full simulation and return the JSON report + chart path.

    Pipeline: seed (TF-IDF keywords + sentiment bias) -> environment
    (small-world graph + agents) -> simulate (round-by-round updates) ->
    evaluate (variance / convergence / clusters) -> report (PNG + JSON).

    Returns:
        A dict mirroring the run's report.json, plus a run_id and
        elapsed_seconds for a quick sanity-check on latency.
    """
    start = time.monotonic()

    _env, log, seed_info = run_simulation(
        topic=request.topic,
        num_agents=request.num_agents,
        num_rounds=request.num_rounds,
    )
    metrics = evaluate(topic=request.topic, seed_info=seed_info, log=log)

    run_id = f"{int(time.time() * 1000)}"
    run_dir = OUTPUT_DIR / run_id
    chart_path = generate_chart(log, run_dir / "chart.png")
    report_path = generate_report(
        topic=request.topic,
        params={"num_agents": request.num_agents, "num_rounds": request.num_rounds},
        seed_info=seed_info,
        metrics=metrics,
        chart_path=chart_path,
        out_json_path=run_dir / "report.json",
    )

    elapsed = time.monotonic() - start

    return {
        "run_id": run_id,
        "topic": request.topic,
        "params": {"num_agents": request.num_agents, "num_rounds": request.num_rounds},
        "seed": seed_info,
        "metrics": metrics,
        "chart_path": str(chart_path),
        "report_path": str(report_path),
        "elapsed_seconds": elapsed,
    }


@app.get("/")
def root() -> dict:
    """Trivial health/info endpoint pointing callers at the real one."""
    return {"service": "SwarmSim", "endpoint": "POST /simulate"}
