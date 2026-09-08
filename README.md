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
  main.py          FastAPI POST /simulate ties it all together
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
| `main.py` | HTTP plumbing | none of the above internals |

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
models involved:

- **Variance per round** — population variance of `opinion_score` across
  all agents at each round. `0` = total agreement, larger = still split.
- **Convergence round** — the first round where variance drops below a
  threshold (default `0.01`) **and stays below it through the end of the
  run**. Requiring it to hold prevents a single noisy dip from being
  reported as convergence.
- **Number of opinion clusters** — connected components of a graph where
  two agents are linked if their *final* opinions are within `eps`
  (default `0.15`) of each other. This sidesteps having to pick `k` up
  front the way k-means would: a chain of mutually-close agents collapses
  into one cluster even if its two ends are far apart, which is exactly
  the "camp" structure we're trying to detect.
- **Text summary** — a formatted string built directly from the numbers
  above (not generated text) so it's always consistent with the report.

## Sample run

Actual output from this repo, topic `"new company policy on remote work"`,
30 agents, 30 rounds, default parameters:

```
seed:
  keywords: ['company', 'new', 'policy', 'remote', 'work']
  bias: 0.00   (none of these keywords are in the sentiment lexicon -> neutral prior)

metrics:
  initial_variance (round 0):  0.0443
  final_variance   (round 30): 0.0016
  convergence_round: 3
  num_clusters: 1

  variance every 5 rounds:
    round  0: 0.0443
    round  5: 0.0075
    round 10: 0.0032
    round 15: 0.0043
    round 20: 0.0020
    round 25: 0.0019
    round 30: 0.0016

summary: "Topic 'new company policy on remote work' started from a
  neutral keyword bias of 0.00 (keywords: company, new, policy, remote,
  work). Opinion variance moved from 0.044 at round 0 to 0.002 at the
  final round, converging (variance < 0.01) at round 3. The swarm settled
  into 1 distinct opinion cluster(s) by the end of the run."
```

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

The API is now at `http://127.0.0.1:8000`.

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
summary) plus `chart_path` / `report_path` pointing at the saved PNG and
`report.json` under `output/<run_id>/`.

### Running the tests

```bash
pytest
```

19 tests cover `agent.py` (the update formula and its edge cases),
`environment.py` (graph construction and the synchronous-step
invariant), and `evaluate.py` (variance, convergence, and clustering
against hand-built logs).

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
  when, into how many clusters), which is exactly the shape of an eval
  harness for a more complex agent system: run, log, score, summarize.

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
