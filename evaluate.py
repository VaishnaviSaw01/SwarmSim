"""
evaluate.py -- turns a round-by-round opinion log into a small set of
benchmarking metrics, plus a human-readable summary.

WHY these particular metrics: they are the standard "did the population
converge, which way, and into how many camps" questions you'd ask of any
opinion dynamics simulation, and each one is computable with nothing more
exotic than variance, a threshold, and a distance-based graph. There is
no fitted model and no hidden scoring anywhere in this file -- every
number a caller sees can be recomputed by hand from the log.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

# The one threshold this file leans on twice: two agents within this
# distance of each other count as "the same camp" (clustering), and one
# agent this far from zero counts as having a lean (distribution). Reusing
# it rather than introducing a second magic number keeps "close enough"
# meaning one thing throughout the module.
LEAN_THRESHOLD = 0.15


def variance_per_round(log: list[dict]) -> dict[int, float]:
    """Population variance of opinion_score at each round.

    WHY variance specifically: it's the simplest single number that
    captures "how spread out is the swarm's opinion right now" -- 0 means
    everyone agrees exactly, large means the swarm is still split.
    """
    by_round: dict[int, list[float]] = defaultdict(list)
    for row in log:
        by_round[row["round"]].append(row["opinion_score"])

    variances = {}
    for round_num, scores in by_round.items():
        mean = sum(scores) / len(scores)
        variances[round_num] = sum((s - mean) ** 2 for s in scores) / len(scores)
    return variances


def find_convergence_round(variances: dict[int, float], threshold: float = 0.01) -> int | None:
    """First round at which variance drops below `threshold` and stays
    there for the rest of the run.

    WHY "and stays there": variance can dip below threshold briefly by
    chance (noise) and pop back up. Requiring it to hold from that round
    through the end of the run is a cheap way to avoid reporting a false
    convergence on a random dip. Returns None if the swarm never
    converges within the logged rounds.
    """
    if not variances:
        return None
    rounds_sorted = sorted(variances)
    for i, r in enumerate(rounds_sorted):
        if all(variances[later] < threshold for later in rounds_sorted[i:]):
            return r
    return None


def _opinion_graph(final_scores: list[float], eps: float) -> nx.Graph:
    """Build the "close enough" graph shared by cluster counting and the
    detailed cluster breakdown, so the two can never disagree about what
    counts as one cluster.

    Two agents (by position in final_scores) are connected if their final
    opinions are within eps of each other.
    """
    g = nx.Graph()
    g.add_nodes_from(range(len(final_scores)))
    for i in range(len(final_scores)):
        for j in range(i + 1, len(final_scores)):
            if abs(final_scores[i] - final_scores[j]) < eps:
                g.add_edge(i, j)
    return g


def count_opinion_clusters(final_scores: list[float], eps: float = 0.15) -> int:
    """Count opinion "camps" in the final round via connected components.

    WHY connected components instead of k-means: k-means needs you to pick
    k up front, which begs the question we're trying to answer ("how many
    clusters are there?"). Instead we build a graph where two agents are
    connected if their final opinions are within `eps` of each other, and
    count connected components -- a chain of agents each within eps of the
    next collapses into one cluster even if the endpoints are far apart,
    which is exactly the "camp" structure we care about, and it needs no
    hyperparameter beyond the same eps we already use for "similar enough".

    Args:
        final_scores: opinion_score of every agent at the last round.
        eps: max opinion distance for two agents to count as connected.

    Returns:
        Number of connected components (>= 1 for a non-empty population).
    """
    if not final_scores:
        return 0
    return nx.number_connected_components(_opinion_graph(final_scores, eps))


def _lean(score: float) -> str:
    """Classify a single opinion score as favorable / opposed / neutral,
    using the shared LEAN_THRESHOLD."""
    if score > LEAN_THRESHOLD:
        return "favorable"
    if score < -LEAN_THRESHOLD:
        return "opposed"
    return "neutral"


def opinion_distribution(final_scores: list[float]) -> dict[str, dict]:
    """What fraction of the swarm ended up favorable / neutral / opposed.

    WHY this on top of variance: variance says *how spread out* the swarm
    is, not *which way* it leans. "68% favorable, 12% opposed, 20%
    neutral" is closer to the sentence a person actually wants out of a
    simulation like this -- it's the same final_scores variance is
    computed from, just bucketed by sign instead of squared.

    Returns:
        {"favorable": {"count": int, "pct": float}, "neutral": {...},
        "opposed": {...}}, percentages rounded to 1 decimal place.
    """
    n = len(final_scores)
    counts = {"favorable": 0, "neutral": 0, "opposed": 0}
    for score in final_scores:
        counts[_lean(score)] += 1

    if n == 0:
        return {lean: {"count": 0, "pct": 0.0} for lean in counts}
    return {lean: {"count": c, "pct": round(100 * c / n, 1)} for lean, c in counts.items()}


def cluster_breakdown(final_scores: list[float], eps: float = 0.15) -> list[dict]:
    """Detail on each opinion cluster, not just how many there are.

    Built from the exact same "close enough" graph as
    count_opinion_clusters (via _opinion_graph), so the two never
    disagree about what counts as one cluster -- this just reports more
    about each component.

    Returns:
        Clusters sorted largest first, each a dict with size, pct (of the
        whole population), mean_score, and lean (favorable/opposed/neutral
        of that cluster's mean).
    """
    if not final_scores:
        return []

    g = _opinion_graph(final_scores, eps)
    n = len(final_scores)
    clusters = []
    for component in nx.connected_components(g):
        scores = [final_scores[i] for i in component]
        mean_score = sum(scores) / len(scores)
        clusters.append(
            {
                "size": len(component),
                "pct": round(100 * len(component) / n, 1),
                "mean_score": mean_score,
                "lean": _lean(mean_score),
            }
        )

    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters


def classify_outcome(clusters: list[dict]) -> str:
    """A one/two-word verdict on the shape of the outcome.

    Not a new computation -- just a plain-language label for a pattern
    already sitting in the cluster breakdown:

        - one cluster                                -> "Consensus"
        - 2+ clusters, top two each >= 25% of the
          population                                   -> "Polarized"
        - anything else (one dominant cluster plus
          smaller holdout groups)                       -> "Majority with holdouts"
    """
    if not clusters:
        return "No data"
    if len(clusters) == 1:
        return "Consensus"
    if clusters[0]["pct"] >= 25 and clusters[1]["pct"] >= 25:
        return "Polarized"
    return "Majority with holdouts"


def swing_agents(log: list[dict]) -> dict | None:
    """Which agent moved the most from round 0 to the final round, and
    which barely moved at all.

    WHY: the aggregate numbers (variance, clusters, distribution)
    describe the population; this gives the run a concrete, checkable
    detail -- one specific agent you could point to. Computed directly
    from round-0 vs. final-round scores per agent_id; no new state.

    Returns:
        {"most_persuaded": {...}, "most_steadfast": {...}} where each
        value has agent_id, from, to, movement (to - from) -- or None if
        the log is empty.
    """
    if not log:
        return None

    first_round = min(row["round"] for row in log)
    last_round = max(row["round"] for row in log)
    first_scores = {row["agent_id"]: row["opinion_score"] for row in log if row["round"] == first_round}
    last_scores = {row["agent_id"]: row["opinion_score"] for row in log if row["round"] == last_round}

    movement = {
        agent_id: last_scores[agent_id] - first_scores[agent_id]
        for agent_id in first_scores
        if agent_id in last_scores
    }
    if not movement:
        return None

    most_persuaded_id = max(movement, key=lambda a: abs(movement[a]))
    most_steadfast_id = min(movement, key=lambda a: abs(movement[a]))

    def _describe(agent_id: int) -> dict:
        return {
            "agent_id": agent_id,
            "from": first_scores[agent_id],
            "to": last_scores[agent_id],
            "movement": movement[agent_id],
        }

    return {
        "most_persuaded": _describe(most_persuaded_id),
        "most_steadfast": _describe(most_steadfast_id),
    }


def summarize(topic: str, seed_info: dict, metrics: dict) -> str:
    """Build a short human-readable summary of the run's outcome.

    This is plain string formatting, not text generation -- every number
    in the summary is read straight out of `metrics`, so the summary is
    always consistent with the numeric report sitting next to it.
    """
    conv = metrics["convergence_round"]
    conv_str = f"round {conv}" if conv is not None else "no round (did not converge)"
    bias_word = (
        "favorable" if seed_info["bias"] > 0.05
        else "unfavorable" if seed_info["bias"] < -0.05
        else "neutral"
    )
    dist = metrics["distribution"]

    swing_sentence = ""
    swing = metrics.get("swing")
    if swing:
        mp, ms = swing["most_persuaded"], swing["most_steadfast"]
        swing_sentence = (
            f" The biggest mover was agent #{mp['agent_id']} ({mp['from']:.2f} -> "
            f"{mp['to']:.2f}); the most steadfast, agent #{ms['agent_id']}, moved "
            f"only {ms['movement']:+.2f}."
        )

    return (
        f"Topic '{topic}' started from a {bias_word} keyword bias of "
        f"{seed_info['bias']:.2f} (keywords: {', '.join(seed_info['keywords']) or 'none'}). "
        f"Opinion variance moved from {metrics['initial_variance']:.3f} at round 0 to "
        f"{metrics['final_variance']:.3f} at the final round, converging (variance < "
        f"{metrics['threshold']}) at {conv_str}. The swarm settled into "
        f"{metrics['num_clusters']} opinion cluster(s) -- {dist['favorable']['pct']}% "
        f"favorable, {dist['opposed']['pct']}% opposed, {dist['neutral']['pct']}% "
        f"neutral -- an outcome best described as \"{metrics['verdict']}\"."
        f"{swing_sentence}"
    )


def evaluate(
    topic: str,
    seed_info: dict,
    log: list[dict],
    threshold: float = 0.01,
    cluster_eps: float = 0.15,
) -> dict:
    """Compute the full evaluation report for one simulation run.

    Args:
        topic: the original topic string (used only for the summary text).
        seed_info: {"keywords", "bias"} as returned by simulate.run_simulation.
        log: the round-by-round log from simulate.run_simulation.
        threshold: variance threshold used for convergence detection.
        cluster_eps: opinion-distance threshold used for cluster counting.

    Returns:
        A dict with variance_per_round, convergence_round, num_clusters,
        clusters (per-cluster detail), distribution (favorable/opposed/
        neutral split), verdict (plain-language outcome label), swing
        (most-moved / least-moved agent), initial_variance,
        final_variance, threshold, and a text summary -- this is exactly
        what report.py writes out as JSON and what main.py returns.
    """
    variances = variance_per_round(log)
    conv_round = find_convergence_round(variances, threshold=threshold)

    last_round = max(row["round"] for row in log)
    final_scores = [row["opinion_score"] for row in log if row["round"] == last_round]

    clusters = cluster_breakdown(final_scores, eps=cluster_eps)

    metrics = {
        "variance_per_round": variances,
        "convergence_round": conv_round,
        "num_clusters": len(clusters),
        "clusters": clusters,
        "distribution": opinion_distribution(final_scores),
        "verdict": classify_outcome(clusters),
        "swing": swing_agents(log),
        "initial_variance": variances[0],
        "final_variance": variances[last_round],
        "threshold": threshold,
        "cluster_eps": cluster_eps,
    }
    metrics["summary"] = summarize(topic, seed_info, metrics)
    return metrics
