# SwarmSim

A lightweight multi-agent opinion simulation engine, built as an
**environment engineering** exercise: define an environment (agents +
graph + update rule), run it, generate structured data, and evaluate the
outcome — all offline, with no paid APIs, no external LLM calls, and no
API keys.

You seed a swarm of agents with an opinion bias derived from a topic
string, drop them onto a small-world social graph, let them influence
their neighbors round by round, and measure whether/when/how the
population converges.

## Why this exists

This isn't trying to be a state-of-the-art opinion dynamics model. Every
piece of it — the update rule, the sentiment scoring, the clustering — is
a formula simple enough to put on a whiteboard. The interesting part is
the *environment*: a reproducible pipeline that turns unstructured text
into simulation parameters, runs a stateful multi-agent system, and
produces both raw data (a round-by-round log) and evaluated metrics
(variance, convergence, clusters) from it.

## Architecture

```
topic string
     |
     v
  seed.py          TF-IDF keywords -> lexicon-based sentiment bias
     |
     v
environment.py     watts_strogatz small-world graph + Agent per node,
     |              opinions seeded around the bias
     v
simulate.py        round loop: environment.step() each round,
     |              logs every agent's opinion_score every round
     v
evaluate.py        variance per round, convergence round,
     |              opinion-cluster count, text summary
     v
 report.py         matplotlib PNG chart + JSON report to disk
     |
     v
  main.py          FastAPI POST /simulate ties it all together,
                    serves static/index.html as the demo UI
```

Each module owns exactly one responsibility, and none of them know about
the layers two steps away:

| Module | Owns | Does NOT know about |
|---|---|---|
| `agent.py` | one agent's state + its update rule | the graph, the topic, other agents |
| `environment.py` | graph topology, running one round | the topic string, evaluation |
| `seed.py` | text -> keywords -> bias | the graph, agents |
| `simulate.py` | wiring seed + environment across rounds | evaluation, reporting |
| `evaluate.py` | metrics over a log | how the log was produced |
| `report.py` | rendering a log/metrics to disk | how metrics were computed |
| `ratelimit.py` | per-client request counting | HTTP, FastAPI, what it's protecting |
| `main.py` | HTTP plumbing, CORS, rate limiting, logging | none of the above internals |
| `static/index.html` | a browser form for `/simulate` | the pipeline internals — talks to it only over HTTP |

That separation is what makes each piece independently testable (see
`tests/`) and independently explainable.

## The interaction rule

Every agent updates once per round as a **stubbornness-weighted average of
its graph neighbors' current opinions, plus Gaussian noise**:

```
new_opinion = w * old_opinion + (1 - w) * mean(neighbor_opinions) + noise
```

- `w` = the agent's `stubbornness` trait, in `[0.1, 0.9]`. `w = 1` would
  mean the agent never changes; `w = 0` means it fully adopts its
  neighbors' average every round.
- `mean(neighbor_opinions)` = the average `opinion_score` of the agent's
  direct neighbors in the social graph, from the round *before* this
  update (see "why synchronous updates" below).
- `noise ~ N(0, noise_std * openness)` — a single Gaussian draw per agent
  per round. `noise_std` is a global run parameter (default `0.05`);
  `openness` is a per-agent trait in `[0.1, 1.0]` that scales it, so more
  "open" agents pick up more randomness.
- The result is clipped to `[-1, 1]`.

**Why synchronous, not sequential, updates:** every agent's update for
round `r` uses a *snapshot* of all opinions from round `r-1`, taken before
any agent in round `r` changes. If agents updated one at a time in graph
order instead, whichever agent happened to be processed first would leak
its brand-new opinion into its neighbors' averages within the same round —
the outcome would then depend on arbitrary node iteration order rather
than on the graph structure itself. Snapshotting makes "one round" mean
the same thing for every agent.

**Why a small-world graph:** `networkx.watts_strogatz_graph(n, k, p)`
starts every agent connected to its `k` nearest neighbors in a ring, then
rewires each edge with probability `p`. That gives local clustering (most
influence comes from nearby neighbors) plus a few long-range shortcuts
(rewired edges), which is closer to a real social network than either a
fully-connected graph (everyone equally influences everyone) or a plain
ring (no shortcuts, information only spreads locally, slowly).

**Seeding the initial opinions:** `seed.py` extracts the topic's top-5
TF-IDF keywords, then scores them against a small hardcoded
positive/negative word list:

```
bias = (positive_keyword_hits - negative_keyword_hits) / num_keywords
```

Every agent's opinion at round 0 is then `bias + N(0, 0.3)`, clipped to
`[-1, 1]` — agents start scattered around the topic's bias rather than all
starting identical, since a zero-variance population at round 0 would
leave nothing for the simulation to converge *from*.

## Evaluation metrics

All computed in `evaluate.py` from the round-by-round log, no fitted
models or LLM calls involved. `LEAN_THRESHOLD` (`0.15`) is the one number
the module leans on twice — reused rather than inventing a second magic
constant:

- **Variance per round** — population variance of `opinion_score` across
  all agents at each round. `0` = total agreement, larger = still split.
- **Convergence round** — the first round where variance drops below a
  threshold (default `0.01`) **and stays below it through the end of the
  run**. Requiring it to hold prevents a single noisy dip from being
  reported as convergence.
- **Opinion clusters** — connected components of a graph where two agents
  are linked if their *final* opinions are within `LEAN_THRESHOLD` of each
  other. This sidesteps having to pick `k` up front the way k-means
  would: a chain of mutually-close agents collapses into one cluster even
  if its two ends are far apart, which is exactly the "camp" structure
  we're trying to detect. `cluster_breakdown()` reports each cluster's
  size, share of the population, mean opinion, and lean (favorable /
  opposed / neutral).
- **Distribution** — `opinion_distribution()` buckets every agent's final
  score as favorable (`> 0.15`), opposed (`< -0.15`), or neutral, and
  reports the count/percentage in each bucket. This answers "which way,
  and how many" — the question variance alone can't.
- **Verdict** — `classify_outcome()` turns the cluster breakdown into a
  one-line label: `Consensus` (one cluster), `Polarized` (2+ clusters,
  the two largest each ≥25% of the population), or `Majority with
  holdouts` (one dominant cluster plus smaller holdout groups). Not a new
  computation — just a name for a shape already in the data.
- **Swing agents** — `swing_agents()` finds the single agent that moved
  most from round 0 to the final round, and the one that moved least: a
  concrete, checkable detail alongside the aggregate numbers.
- **Text summary** — a formatted string built directly from the numbers
  above (not generated text) so it's always consistent with the report.

## Sample run

Actual output from this repo, topic `"new company policy on remote work"`,
30 agents, 30 rounds, default parameters (`seed=42`, so this exact run is
reproducible):

```
seed:
  keywords: ['company', 'new', 'policy', 'remote', 'work']
  bias: 0.00   (none of these keywords are in the sentiment lexicon -> neutral prior)

metrics:
  initial_variance (round 0):  0.0443
  final_variance   (round 30): 0.0016
  convergence_round: 3
  verdict: "Consensus"
  distribution: favorable 0.0% (0), neutral 100.0% (30), opposed 0.0% (0)
  clusters: [{size: 30, pct: 100.0, mean_score: -0.066, lean: "neutral"}]
  swing:
    most_persuaded: agent #2,  -0.50 -> 0.02  (moved +0.53)
    most_steadfast: agent #28, -0.14 -> -0.12 (moved +0.02)

summary: "Topic 'new company policy on remote work' started from a
  neutral keyword bias of 0.00 (keywords: company, new, policy, remote,
  work). Opinion variance moved from 0.044 at round 0 to 0.002 at the
  final round, converging (variance < 0.01) at round 3. The swarm settled
  into 1 opinion cluster(s) -- 0.0% favorable, 0.0% opposed, 100.0%
  neutral -- an outcome best described as "Consensus". The biggest mover
  was agent #2 (-0.50 -> 0.02); the most steadfast, agent #28, moved only
  +0.02."
```

Run the negatively-loaded topic from the curl examples below
(`"mandatory return to office is a burden and unfair surveillance"`,
bias -0.60) and the same population instead lands on `verdict:
"Consensus"` with `opposed: 100%` — the swarm still agrees, just in the
other direction. That favorable/opposed split, not just the variance
number, is what the demo UI's distribution bar is showing.

Chart description (`chart.png`, 50 agents / 50 rounds run): agents start
at round 0 scattered between roughly -0.65 and +0.9 around the neutral
bias; lines fan in sharply over the first ~15 rounds; by round ~30 all 50
trajectories have tightened into a narrow band clustered around ~-0.1,
with the small residual spread coming entirely from the per-round Gaussian
noise term rather than genuine disagreement — matching `num_clusters: 1`.

Full pipeline (seed → environment → simulate → evaluate → chart + JSON
report) for **50 agents / 50 rounds measured on this machine: 0.24s** —
well inside the 5-second budget.

## Running it

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/` for the demo UI — a hero with clickable
example topics, a 4-step "how it works" walkthrough, the try-it form, and
a results view with a plain-language verdict, a favorable/neutral/opposed
distribution bar, per-cluster cards, the most-persuaded/most-steadfast
agent, stat tiles, the trajectory chart, and a raw-JSON toggle — or use
the API directly, with interactive docs at `http://127.0.0.1:8000/docs`.

### Example requests

Neutral topic, small swarm, quick run:

```bash
curl -X POST http://127.0.0.1:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"topic": "new company policy on remote work", "num_agents": 30, "num_rounds": 30}'
```

Positively-loaded topic (keyword bias comes out around **+0.6**):

```bash
curl -X POST http://127.0.0.1:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"topic": "flexible remote work improves productivity and trust", "num_agents": 40, "num_rounds": 40}'
```

Negatively-loaded topic (keyword bias comes out around **-0.6**):

```bash
curl -X POST http://127.0.0.1:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"topic": "mandatory return to office is a burden and unfair surveillance", "num_agents": 40, "num_rounds": 40}'
```

Larger swarm, more rounds, still fast:

```bash
curl -X POST http://127.0.0.1:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"topic": "quarterly performance review process", "num_agents": 100, "num_rounds": 60}'
```

Each call returns a JSON report (topic, params, seed info, metrics,
summary) plus `chart_data_url` — the chart PNG inline as base64, ready to
drop straight into an `<img src="...">` — and, best-effort, `chart_url` /
`report_path` pointing at a local copy on the server (not guaranteed on
every host — see "Deployment" below). `num_agents` / `num_rounds` are
capped at 200 each on this endpoint — see "Deployment" for why.

### Running the tests

```bash
pytest
```

40 tests cover `agent.py` (the update formula and its edge cases),
`environment.py` (graph construction and the synchronous-step
invariant), `evaluate.py` (variance, convergence, clustering,
distribution, verdict, and swing agents against hand-built logs),
`report.py` (chart bytes and the disk-writing wrapper), `ratelimit.py`
(the sliding-window counter), and `main.py` (the API end to end: the UI
route, `/simulate` -> inline chart + best-effort `/chart/{id}`,
validation, and rate-limit behavior via `TestClient`).

## Deployment

The chart PNG is delivered inline as base64 in the `/simulate` response
(`chart_data_url`) rather than written to a file and fetched by a second
request — that works identically whether the host keeps one warm process
around (Render, a VPS, Docker) or runs every request as an independent,
stateless invocation with no shared disk (Vercel). Writing a copy to disk
(`OUTPUT_DIR`, default the OS temp dir) is a best-effort side effect for
local inspection, not something any response depends on — see
`GET /chart/{run_id}`'s docstring in [main.py](main.py) for exactly when
that route does and doesn't work.

**Vercel:** zero-config — Vercel's Python runtime auto-detects `main.py`'s
top-level `app` (FastAPI) and `requirements.txt`. Import this repo at
[vercel.com/new](https://vercel.com/new), or `vercel deploy` from the repo
root. `vercel.json` excludes `tests/` etc. from the function bundle but
deliberately *keeps* `static/` in it, since `main.py` reads
`static/index.html` off disk at request time. Free (Hobby plan): 500 MB
Python bundle limit (comfortably fits scikit-learn + matplotlib + scipy),
10s execution timeout (this app's worst case is under 1s).

**Google Cloud Run:** builds and runs this repo's `Dockerfile` as-is (it
already binds `0.0.0.0` and reads the `$PORT` Cloud Run injects at
runtime, so no Cloud-Run-specific changes were needed). Two ways to
deploy:

- *Console, no CLI needed:* Cloud Run -> Create Service -> "Continuously
  deploy from a repository" -> connect this GitHub repo, branch `main` ->
  Cloud Run detects the `Dockerfile` and builds with Cloud Build ->
  every push to `main` triggers a rebuild automatically. Set the env vars
  from the table below under "Variables & Secrets"; leave "Port" as
  whatever Cloud Run auto-detects from the `Dockerfile` (it reads `$PORT`
  itself, so this Just Works). Allow unauthenticated invocations so the
  demo UI is publicly reachable.
- *CLI:* `gcloud run deploy swarmsim --source . --region us-central1
  --allow-unauthenticated` from the repo root (`.gcloudignore` keeps the
  Cloud Build upload small).

Cloud Run's always-free tier (2M requests/month, well beyond what a demo
gets) means this costs $0 as long as usage stays under that quota, which
a portfolio/interview demo will not come close to — but it does require
adding a card to the Google Cloud project for billing verification, and
that verification step can itself fail for reasons outside this repo's
control (issuing bank declines, region restrictions, prior free-trial use).

**Render (this repo's `render.yaml`):** New + -> Blueprint -> point it at
this repo. It builds with `pip install -r requirements.txt` and starts
with `uvicorn main:app --host 0.0.0.0 --port $PORT`, on Render's native
Python runtime (no Docker build there). To do the same by hand in the
Render dashboard instead of using the blueprint: New + -> Web Service,
same build/start commands, health check path `/health`.

**Anywhere else (Railway, a VPS):** this repo's `Dockerfile` —
`docker build -t swarmsim . && docker run -p 8000:8000 -e PORT=8000
swarmsim`.

> Hugging Face Spaces was considered but ruled out: as of 2026, HF
> requires a paid PRO plan just to *create* a Docker or Gradio Space (the
> free "CPU basic" hardware only became free-to-use, not free-to-unlock).
> Its Static SDK stays free for anyone, but Static Spaces serve
> pre-built files only — no Python process — so it can't run this app's
> backend at all.

**Environment variables** (all optional, sensible local defaults):

| Var | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `*` | Comma-separated origins allowed by CORS, or `*` for any |
| `RATE_LIMIT_MAX` | `10` | Max `/simulate` calls per client per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window length |
| `OUTPUT_DIR` | OS temp dir | Where chart PNGs + report.json are (best-effort) written; set to `./output` for local dev if you want them next to the repo |
| `LOG_LEVEL` | `INFO` | Python logging level |

**Public-deployment hardening already in `main.py`:**
- The chart is returned inline as base64 (`chart_data_url`) instead of
  `/simulate` returning a server-local path — works on a stateless
  serverless host with no disk shared between requests, not just a
  single always-on process. `GET /chart/{run_id}` still exists as a
  best-effort convenience where a real filesystem persists across
  requests (`run_id` is validated against a strict digits-only pattern
  before it ever touches the filesystem).
- A per-client sliding-window rate limiter (`ratelimit.py`) sits in front
  of `/simulate`, the one endpoint that does real CPU work.
- `num_agents`/`num_rounds` are capped at 200 (not the 500 the pipeline
  can technically handle) — 200x200 finishes in well under a second,
  while 500x500 takes several seconds of CPU per request, which is a
  cheap way to make this single process fall over if hit repeatedly.
- A global exception handler returns a small stable JSON error and logs
  server-side, instead of leaking a Python traceback to the client.
- Writing the on-disk copy (chart PNG + report.json) is wrapped so a
  read-only or ephemeral filesystem degrades to "no local copy" rather
  than a 500 — the client-facing response never depended on that write
  succeeding.
- `GET /health` for the platform's liveness/health checks.

**Known limitation:** the rate limiter's state is in-process (per
`RateLimiter` instance), so it isn't shared across multiple workers,
instances, or serverless invocations. Fine for a single-process
deployment or for demo-level serverless traffic; would need a shared
store (Redis) to stay correct once traffic is split across many
concurrent workers.

## Interview talking points

**What this demonstrates:**
- *Environment design* — a small, composable state machine (agents +
  graph + update rule) with a single well-defined transition function
  (`Environment.step`), built so each layer is independently testable and
  the whole thing is reproducible via one RNG seed threaded through graph
  generation, persona assignment, and per-round noise.
- *Data pipeline* — an explicit, inspectable path from unstructured input
  (a topic string) to structured simulation parameters (keywords, bias) to
  structured output (a long-format round-by-round log), with no step that
  can't be explained by pointing at a formula.
- *Evaluation / benchmarking* — the simulation isn't just run and plotted;
  it's scored against explicit, thresholded criteria (did it converge, by
  when, into how many clusters, which way), which is exactly the shape of
  an eval harness for a more complex agent system: run, log, score,
  summarize.
- *Turning raw numbers into a legible result* — the first pass reported
  variance and a cluster count, which is correct but not a sentence a
  person can act on. `distribution`, `clusters`, `verdict`, and
  `swing_agents` in [evaluate.py](evaluate.py) are all *deterministic
  reprocessing of the same data* (no new model, no LLM), reframed as
  "which way did it lean, by how much, and who moved" — the difference
  between an evaluation harness that produces numbers and one that
  produces an answer someone would actually read.
- *Taking a service from "runs on my machine" to publicly deployable* —
  the one CPU-heavy endpoint is rate-limited, parameter caps are set from
  measured worst-case latency (not guessed), errors are caught and logged
  instead of leaking tracebacks, and config (CORS origins, rate limit,
  output dir) is environment-driven so the same code runs locally and on
  Render without edits.

**What I'd improve with more time:**
- Swap the rule-based `Agent.update_opinion` for an optional "LLM-reaction"
  mode, where an agent's opinion shift comes from a language model judging
  its neighbors' stated positions rather than a plain weighted average —
  the rest of the pipeline (graph, logging, evaluation, reporting) would
  need zero changes, since it only depends on `opinion_score` being a
  float in `[-1, 1]` each round.
- Replace the eps-based clustering with a proper 1D density-based method
  (e.g. `KMeans` with a silhouette-score sweep over `k`, or DBSCAN) once
  cluster shapes get less trivially separable than "gaps in a sorted
  list".
- Add a `/simulate/batch` endpoint to sweep parameters (graph density,
  noise level, initial bias) across many seeded runs and aggregate
  convergence statistics — turning this from "run one simulation" into
  "run an experiment".
- Persist run history (currently just files under `output/`) into
  something queryable, so metrics can be compared across runs instead of
  read one report.json at a time.

**How this relates to RL environment / eval system work:**
- `Environment.step()` is structurally the same contract as an RL
  environment's `step()`: fixed state, one synchronous transition per
  call, deterministic given the RNG seed. Swapping the update rule for a
  learned policy wouldn't touch anything else in the pipeline.
- The seed → run → log → evaluate → report pipeline is the same shape as
  an eval harness for any agentic system: produce a task/scenario,
  execute the agent(s) against it, log every intermediate state, then
  score the trajectory against explicit metrics rather than eyeballing
  the output. Everything downstream of `simulate.run_simulation` (the
  log) doesn't care whether the numbers came from a hand-written formula
  or a model — that's exactly the seam an LLM-based agent mode would slot
  into.
