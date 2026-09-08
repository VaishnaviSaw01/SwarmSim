"""
evaluate.py -- turns a round-by-round opinion log into a small set of
benchmarking metrics, plus a human-readable summary.

WHY these particular metrics: they are the standard "did the population
converge, and into how many camps" questions you'd ask of any opinion
dynamics simulation, and each one is computable with nothing more exotic
than variance, a threshold, and a distance-based graph.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx


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

    g = nx.Graph()
    g.add_nodes_from(range(len(final_scores)))
    for i in range(len(final_scores)):
        for j in range(i + 1, len(final_scores)):
            if abs(final_scores[i] - final_scores[j]) < eps:
                g.add_edge(i, j)

    return nx.number_connected_components(g)


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

    return (
        f"Topic '{topic}' started from a {bias_word} keyword bias of "
        f"{seed_info['bias']:.2f} (keywords: {', '.join(seed_info['keywords']) or 'none'}). "
        f"Opinion variance moved from {metrics['initial_variance']:.3f} at round 0 to "
        f"{metrics['final_variance']:.3f} at the final round, converging (variance < "
        f"{metrics['threshold']}) at {conv_str}. The swarm settled into "
        f"{metrics['num_clusters']} distinct opinion cluster(s) by the end of the run."
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
        initial_variance, final_variance, threshold, and a text summary --
        this is exactly what report.py writes out as JSON.
    """
    variances = variance_per_round(log)
    conv_round = find_convergence_round(variances, threshold=threshold)

    last_round = max(row["round"] for row in log)
    final_scores = [row["opinion_score"] for row in log if row["round"] == last_round]
    num_clusters = count_opinion_clusters(final_scores, eps=cluster_eps)

    metrics = {
        "variance_per_round": variances,
        "convergence_round": conv_round,
        "num_clusters": num_clusters,
        "initial_variance": variances[0],
        "final_variance": variances[last_round],
        "threshold": threshold,
        "cluster_eps": cluster_eps,
    }
    metrics["summary"] = summarize(topic, seed_info, metrics)
    return metrics
