"""
main.py -- FastAPI wrapper around the simulate -> evaluate -> report
pipeline, hardened for public deployment.

WHY a small number of endpoints: this project's interesting surface area
is the simulation pipeline itself (agent.py through report.py), all of
which is plain, synchronous Python you can run and test without a server
at all (see simulate.py and tests/). The FastAPI layer exists to (a)
demonstrate that the pipeline is easy to wrap as a service, and (b) give
it a small, self-contained demo UI (static/index.html) so a public
deployment is something a person can actually click through, not just
curl.

WHAT "hardened" means here, concretely:
  - a per-client rate limiter in front of the one CPU-heavy endpoint
    (ratelimit.py), since this runs as a single process;
  - a global exception handler so an unexpected failure returns a plain
    JSON error instead of a stack trace;
  - CORS, so the API can be called from a frontend on another origin;
  - config (allowed origins, rate limit, output dir) read from
    environment variables with sane local defaults, so nothing needs to
    change between "run on my machine" and "run on Render";
  - request/error logging, since a public service with no logs is a
    public service you can't debug.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from evaluate import evaluate
from ratelimit import RateLimiter
from report import generate_report, render_chart_png
from simulate import run_simulation

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("swarmsim")

app = FastAPI(
    title="SwarmSim",
    description="Multi-agent opinion simulation engine -- offline, no external APIs.",
)

# --- config (env-driven so the same image runs locally and on Render/Cloud
# Run/Vercel without edits) ---
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# Default to the system temp dir, not a folder next to the source code:
# on a serverless host (Vercel) or any read-only deployment filesystem,
# only the OS temp dir is guaranteed writable. Set OUTPUT_DIR explicitly
# (e.g. to "./output") for local development if you want the artifacts
# to land next to the repo instead.
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(tempfile.gettempdir()) / "swarmsim-output"))

_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = ["*"] if _allowed_origins == "*" else [o.strip() for o in _allowed_origins.split(",")]

RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "10"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# --- middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# One limiter instance for the process's lifetime -- see ratelimit.py for
# why this is per-process, not shared across workers.
_simulate_limiter = RateLimiter(max_requests=RATE_LIMIT_MAX, window_seconds=RATE_LIMIT_WINDOW_SECONDS)

_RUN_ID_RE = re.compile(r"^[0-9]{1,20}$")  # run_ids are millisecond epoch timestamps


class SimulateRequest(BaseModel):
    """Request body for POST /simulate.

    num_agents/num_rounds are capped at 200 each (not the 500 the
    underlying pipeline can technically handle) specifically because this
    endpoint is publicly reachable: 200x200 finishes in under a second,
    while 500x500 takes several seconds of solid CPU time per request --
    fine for one call locally, not fine as a repeatable public attack
    surface even with the rate limiter in front of it.
    """

    topic: str = Field(..., min_length=1, max_length=500, description="Topic text to seed opinions from")
    num_agents: int = Field(30, ge=4, le=200, description="Number of agents in the swarm")
    num_rounds: int = Field(30, ge=1, le=200, description="Number of simulation rounds")


@app.post("/simulate")
def simulate(request: SimulateRequest, http_request: Request) -> dict:
    """Run one full simulation and return the JSON report + the chart.

    Pipeline: seed (TF-IDF keywords + sentiment bias) -> environment
    (small-world graph + agents) -> simulate (round-by-round updates) ->
    evaluate (variance / convergence / clusters) -> report (PNG + JSON).

    Rate limited per client IP (see ratelimit.py) because, unlike a normal
    CRUD endpoint, this one does real CPU work.

    Returns:
        A dict with the run_id, params, seed info, metrics, the chart as
        a data: URI (chart_data_url -- works everywhere, including
        serverless hosts with no shared disk between requests),
        chart_url (a best-effort GET route -- see get_chart), and
        elapsed_seconds.
    """
    client_key = http_request.client.host if http_request.client else "unknown"
    allowed, retry_after = _simulate_limiter.allow(client_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before running another simulation.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    start = time.monotonic()
    logger.info(
        "simulate start topic=%r num_agents=%d num_rounds=%d client=%s",
        request.topic, request.num_agents, request.num_rounds, client_key,
    )

    _env, log, seed_info = run_simulation(
        topic=request.topic,
        num_agents=request.num_agents,
        num_rounds=request.num_rounds,
    )
    metrics = evaluate(topic=request.topic, seed_info=seed_info, log=log)

    # Build the chart once, in memory -- this is what actually gets
    # delivered to the client (chart_data_url below), so it has to work
    # even if nothing can be written to disk.
    chart_bytes = render_chart_png(log)
    chart_data_url = f"data:image/png;base64,{base64.b64encode(chart_bytes).decode()}"

    run_id = f"{int(time.time() * 1000)}"
    run_dir = OUTPUT_DIR / run_id

    # Writing the PNG/JSON to disk is a best-effort convenience (local
    # inspection, or a same-instance GET /chart/{run_id}), not something
    # the response depends on -- a read-only deployment filesystem
    # shouldn't turn into a 500 for a request that already succeeded.
    chart_path: Path | None = None
    report_path: Path | None = None
    try:
        chart_path = run_dir / "chart.png"
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        chart_path.write_bytes(chart_bytes)
        report_path = generate_report(
            topic=request.topic,
            params={"num_agents": request.num_agents, "num_rounds": request.num_rounds},
            seed_info=seed_info,
            metrics=metrics,
            chart_path=chart_path,
            out_json_path=run_dir / "report.json",
        )
    except OSError:
        logger.warning("Could not write run artifacts to disk for run_id=%s (non-fatal)", run_id)

    elapsed = time.monotonic() - start
    logger.info("simulate done run_id=%s elapsed=%.3fs", run_id, elapsed)

    return {
        "run_id": run_id,
        "topic": request.topic,
        "params": {"num_agents": request.num_agents, "num_rounds": request.num_rounds},
        "seed": seed_info,
        "metrics": metrics,
        "chart_data_url": chart_data_url,
        "chart_url": f"/chart/{run_id}" if chart_path else None,
        "report_path": str(report_path) if report_path else None,
        "elapsed_seconds": elapsed,
    }


@app.get("/chart/{run_id}")
def get_chart(run_id: str) -> FileResponse:
    """Best-effort: serve a previously generated chart PNG by run_id.

    WHY "best-effort": this reads from local disk, which only exists
    within a single warm process. On a single-instance host (Render,
    Docker, a plain VPS, or Cloud Run at low traffic) it usually works;
    on a serverless platform where each request can land on a different,
    stateless invocation (Vercel), it will often 404 even for a run_id
    that just succeeded. Use chart_data_url from the /simulate response
    for the delivery method that's guaranteed to work everywhere.

    run_id is validated against a strict digits-only pattern (it's always
    a millisecond timestamp we generated) before touching the filesystem,
    so a crafted run_id like "../../etc/passwd" is rejected outright
    rather than reaching Path().
    """
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run_id")

    chart_path = OUTPUT_DIR / run_id / "chart.png"
    if not chart_path.is_file():
        raise HTTPException(status_code=404, detail="Chart not found (try chart_data_url instead)")

    return FileResponse(chart_path, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-page demo UI (static/index.html).

    Kept as a plain file read rather than a templating engine -- the page
    has no server-side variables to inject, so there's nothing a template
    engine would buy here.
    """
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    """Liveness endpoint for uptime checks / the hosting platform."""
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch anything that isn't already an HTTPException.

    WHY: without this, an unexpected error (e.g. a malformed topic string
    tripping up a library call) would surface as a raw Python traceback in
    the HTTP response -- fine for local debugging, a real problem for a
    publicly reachable service (it leaks internals and looks broken
    rather than handled). This logs the full exception server-side and
    returns a small, stable JSON error to the client.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "message": "Something went wrong running the simulation."},
    )
