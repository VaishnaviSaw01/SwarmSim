"""
environment.py -- builds the social graph and advances the swarm one round
at a time.

WHY a small-world graph: real social networks (and most interesting
multi-agent environments) are neither fully connected nor a simple ring --
they have local clustering (your neighbors know each other) plus a handful
of long-range shortcuts (a few "bridge" connections between otherwise
distant clusters). networkx's watts_strogatz_graph gives us exactly that
with two knobs (k = local connectivity, p = rewiring probability) instead
of hand-building a custom topology. It's a standard, well-documented
generator -- not a hidden black box -- which matters for a project whose
whole point is to be explainable end to end.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from agent import Agent


class Environment:
    """Owns the social graph and the collection of agents living on it.

    The Environment is the only object that knows about graph topology.
    Agents don't know who their neighbors are; the Environment looks that
    up each round and hands each agent only the numbers it needs (its
    neighbors' opinion scores). This keeps Agent.update_opinion pure and
    testable in isolation from the graph entirely.
    """

    def __init__(
        self,
        num_agents: int,
        k: int = 4,
        rewire_prob: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Build the graph and instantiate one Agent per node.

        Args:
            num_agents: number of agents / graph nodes.
            k: each node starts joined to its k nearest neighbors in a ring
                topology. Must be even and less than num_agents; adjusted
                automatically if it isn't (see _resolve_k).
            rewire_prob: probability that a given ring edge is rewired to a
                random node -- this is what creates the "small-world"
                long-range shortcuts.
            seed: RNG seed, threaded through both the graph generator and
                persona/opinion randomness so a whole run is reproducible.
        """
        self.rng = np.random.default_rng(seed)
        resolved_k = self._resolve_k(num_agents, k)
        self.graph = nx.watts_strogatz_graph(n=num_agents, k=resolved_k, p=rewire_prob, seed=seed)
        self.agents: dict[int, Agent] = {
            node: Agent(agent_id=node, persona=self._random_persona())
            for node in self.graph.nodes
        }

    @staticmethod
    def _resolve_k(num_agents: int, k: int) -> int:
        """Clamp k into a value watts_strogatz_graph will accept: even,
        at least 2, and strictly less than num_agents."""
        if k % 2 != 0:
            k += 1
        k = min(k, num_agents - 1)
        if k % 2 != 0:
            k -= 1
        return max(k, 2)

    def _random_persona(self) -> dict[str, float]:
        """Draw a random persona for a new agent.

        Two traits, each uniform in [0, 1], is deliberately minimal: enough
        for agents to behave differently from one another (see
        Agent.update_opinion) without needing a persona *model*.
        """
        return {
            "stubbornness": float(self.rng.uniform(0.1, 0.9)),
            "openness": float(self.rng.uniform(0.1, 1.0)),
        }

    def seed_opinions(self, bias: float, sigma: float = 0.3) -> None:
        """Initialize every agent's opinion_score around a shared bias.

        WHY: seed.py computes a single scalar "bias" for the topic (from
        keyword sentiment). Real people don't all start at exactly the same
        opinion even when reacting to the same news, so we scatter agents
        around that bias with Gaussian noise rather than pinning every
        agent to the identical value -- otherwise round 0 would already be
        a degenerate, zero-variance population and there'd be nothing for
        the simulation to converge *from*.
        """
        for agent in self.agents.values():
            noise = float(self.rng.normal(0, sigma))
            agent.opinion_score = max(-1.0, min(1.0, bias + noise))

    def step(self, round_num: int, noise_std: float = 0.05) -> dict[int, float]:
        """Advance the whole swarm by one round.

        WHY synchronous updates: we snapshot every agent's opinion_score
        *before* any agent updates, then apply all updates from that one
        snapshot. If we updated agents one at a time in graph node order,
        agent 5's neighbors could already reflect round r+1 while agent 2's
        neighbors still reflect round r -- the outcome would depend on
        arbitrary node ordering. Snapshotting makes one "round" mean the
        same thing for every agent, which is what you want from an
        environment used to generate comparable simulation data.

        Args:
            round_num: index of this round (recorded into agent memory).
            noise_std: base standard deviation of the per-agent Gaussian
                noise, scaled per-agent by that agent's "openness" trait.

        Returns:
            Mapping of agent_id -> new opinion_score after this round.
        """
        snapshot = {aid: agent.opinion_score for aid, agent in self.agents.items()}

        new_scores: dict[int, float] = {}
        for aid, agent in self.agents.items():
            neighbor_ids = list(self.graph.neighbors(aid))
            neighbor_scores = [snapshot[n] for n in neighbor_ids]
            openness = agent.persona.get("openness", 0.5)
            noise = float(self.rng.normal(0, noise_std * openness))
            new_scores[aid] = agent.update_opinion(neighbor_scores, noise, round_num)

        return new_scores

    def opinion_scores(self) -> dict[int, float]:
        """Current opinion_score of every agent, keyed by agent_id."""
        return {aid: agent.opinion_score for aid, agent in self.agents.items()}
