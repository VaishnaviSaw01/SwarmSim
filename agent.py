"""
agent.py -- defines the Agent, the atomic unit of the swarm.

WHY this exists: every multi-agent simulation needs a minimal, inspectable
unit of state. We deliberately keep Agent "dumb" -- it holds state and
knows how to update itself given its neighbors' opinions, but it has no
knowledge of the graph, the topic, or the rest of the swarm. That
separation is what lets environment.py own the *topology* and simulate.py
own the *schedule*, while agent.py owns only the *update rule*. Keeping
agents ignorant of everything except their own history makes the whole
system easy to reason about and to explain on a whiteboard.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Agent:
    """A single opinionated agent in the swarm.

    Attributes:
        agent_id: unique identifier, also the node id in the social graph.
        persona: a small dict of traits in [0, 1] that shape how the agent
            reacts to its neighbors. Kept to 2-3 traits on purpose -- enough
            to make agents heterogeneous without turning the update rule
            into something you can no longer derive by hand:
                - "stubbornness": how much weight the agent puts on its own
                  current opinion vs. its neighbors' average (see
                  `update_opinion`).
                - "openness": scales how much random noise the agent's
                  opinion picks up each round (a proxy for "how easily
                  swayed by things outside the model").
        opinion_score: the agent's current opinion, constrained to [-1, 1]
            (-1 = fully against, +1 = fully in favor, 0 = neutral).
        memory: the agent's last 5 interactions, each a small dict recording
            what round it was and what it observed from its neighborhood.
            This is *not* consulted by the update rule -- it exists so the
            agent carries an auditable trace of "why did my opinion move",
            which is useful for debugging and demonstrates this is a real
            stateful agent rather than a bare number sitting on a graph.
    """

    agent_id: int
    persona: dict[str, float]
    opinion_score: float = 0.0
    memory: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5))

    def __post_init__(self) -> None:
        self.opinion_score = _clip(self.opinion_score)

    def update_opinion(
        self,
        neighbor_scores: list[float],
        noise: float,
        round_num: int,
    ) -> float:
        """Update this agent's opinion given its neighbors' current scores.

        WHY a weighted average + noise, and nothing fancier: the point of
        this project is an *environment*, not a model. A rule that can be
        written on a whiteboard and defended line-by-line in an interview
        is more valuable here than a rule that "performs better" but is
        opaque. The formula is:

            new_opinion = w * old_opinion + (1 - w) * mean(neighbor_scores) + noise

        where `w` is this agent's "stubbornness" trait (0 = fully swayed by
        neighbors, 1 = completely ignores them) and `noise` is a single
        Gaussian draw, pre-scaled by the caller using the agent's
        "openness" trait so that more "open" agents pick up more
        randomness per round.

        If the agent has no neighbors (an isolated node), there is nothing
        to average against, so it keeps its own score and only absorbs the
        noise term.

        Args:
            neighbor_scores: current opinion_score of every neighbor in the
                social graph, *before* this round's updates are applied
                (see Environment.step for why that ordering matters --
                updates are synchronous, not sequential).
            noise: a single pre-drawn Gaussian sample for this agent this
                round (drawn by the caller so the RNG stream stays
                centrally controlled and reproducible).
            round_num: the current round index, recorded into memory.

        Returns:
            The agent's new opinion_score (also stored on the instance).
        """
        w = self.persona.get("stubbornness", 0.5)
        neighbor_avg = (
            sum(neighbor_scores) / len(neighbor_scores)
            if neighbor_scores
            else self.opinion_score
        )

        new_score = _clip(w * self.opinion_score + (1 - w) * neighbor_avg + noise)

        self.memory.append(
            {
                "round": round_num,
                "neighbor_avg": neighbor_avg,
                "num_neighbors": len(neighbor_scores),
                "old_score": self.opinion_score,
                "new_score": new_score,
            }
        )
        self.opinion_score = new_score
        return new_score

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent's public state (used by logging/report code)."""
        return {
            "agent_id": self.agent_id,
            "persona": dict(self.persona),
            "opinion_score": self.opinion_score,
        }


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp a value into [lo, hi].

    Opinions are bounded by definition, so a plain min/max clamp is all we
    need -- no sigmoid or other squashing function required.
    """
    return max(lo, min(hi, value))
