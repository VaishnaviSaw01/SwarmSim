"""
simulate.py -- orchestrates a full run: seed the swarm, then advance it
round by round, logging every agent's opinion at every round.

WHY a flat list-of-dicts log instead of a nested structure: it maps
directly onto a "long" table (round, agent_id, opinion_score), which is
the easiest shape to compute statistics over (evaluate.py) and to plot
(report.py) without any reshaping step in between.
"""

from __future__ import annotations

from environment import Environment
from seed import seed_from_topic


def run_simulation(
    topic: str,
    num_agents: int,
    num_rounds: int,
    k: int = 4,
    rewire_prob: float = 0.1,
    noise_std: float = 0.05,
    seed: int | None = 42,
) -> tuple[Environment, list[dict], dict]:
    """Run one full simulation from a topic string to a round-by-round log.

    Args:
        topic: raw topic text, e.g. "new company policy on remote work".
        num_agents: swarm size.
        num_rounds: how many rounds to simulate (round 0 is the seeded
            initial state, logged before any update runs).
        k, rewire_prob: small-world graph parameters, see Environment.
        noise_std: base Gaussian noise std for Agent.update_opinion.
        seed: RNG seed for reproducibility (graph, personas, noise).

    Returns:
        (environment, log, seed_info) where:
            - environment: the final Environment, in case a caller wants
              direct access to the graph or agents.
            - log: list of {"round", "agent_id", "opinion_score"} dicts,
              one row per agent per round, including round 0.
            - seed_info: {"keywords": [...], "bias": float} from seed.py,
              kept separate from the per-round log since it describes the
              *input*, not a simulation round.
    """
    keywords, bias = seed_from_topic(topic)

    env = Environment(num_agents=num_agents, k=k, rewire_prob=rewire_prob, seed=seed)
    env.seed_opinions(bias=bias)

    log: list[dict] = []

    def _log_round(round_num: int, scores: dict[int, float]) -> None:
        for agent_id, score in scores.items():
            log.append({"round": round_num, "agent_id": agent_id, "opinion_score": score})

    _log_round(0, env.opinion_scores())

    for round_num in range(1, num_rounds + 1):
        scores = env.step(round_num=round_num, noise_std=noise_std)
        _log_round(round_num, scores)

    seed_info = {"keywords": keywords, "bias": bias}
    return env, log, seed_info
