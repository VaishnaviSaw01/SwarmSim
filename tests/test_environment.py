"""Tests for environment.py -- graph construction and one round of updates."""

from environment import Environment


def test_graph_has_correct_number_of_agents():
    env = Environment(num_agents=20, k=4, rewire_prob=0.1, seed=1)
    assert len(env.agents) == 20
    assert env.graph.number_of_nodes() == 20


def test_agents_have_neighbors():
    env = Environment(num_agents=20, k=4, rewire_prob=0.1, seed=1)
    for node in env.graph.nodes:
        assert env.graph.degree[node] > 0


def test_resolve_k_handles_small_populations():
    # num_agents=5 with a requested k=8 must be clamped below num_agents.
    env = Environment(num_agents=5, k=8, rewire_prob=0.1, seed=1)
    assert env.graph.number_of_nodes() == 5


def test_seed_opinions_sets_scores_near_bias():
    env = Environment(num_agents=50, k=4, rewire_prob=0.1, seed=1)
    env.seed_opinions(bias=0.5, sigma=0.0)  # sigma=0 -> deterministic
    scores = env.opinion_scores()
    assert all(score == 0.5 for score in scores.values())


def test_step_changes_opinion_scores():
    env = Environment(num_agents=30, k=4, rewire_prob=0.1, seed=1)
    env.seed_opinions(bias=0.0, sigma=0.5)
    before = dict(env.opinion_scores())

    env.step(round_num=1, noise_std=0.05)
    after = env.opinion_scores()

    assert before.keys() == after.keys()
    assert any(before[aid] != after[aid] for aid in before)


def test_step_is_synchronous_not_order_dependent():
    """Regression test for the snapshot-before-update behavior described in
    Environment.step's docstring: an agent's update must use its
    neighbors' *pre-round* scores, not scores already updated this round."""
    env = Environment(num_agents=10, k=2, rewire_prob=0.0, seed=2)
    for agent in env.agents.values():
        agent.persona["stubbornness"] = 0.0  # fully swayed by neighbors

    env.seed_opinions(bias=0.0, sigma=0.0)
    env.agents[0].opinion_score = 1.0  # perturb one agent away from the rest

    env.step(round_num=1, noise_std=0.0)

    # Agent 0's direct neighbors should reflect agent 0's *old* score (1.0)
    # mixed into their average, not some already-updated intermediate value.
    neighbor_ids = list(env.graph.neighbors(0))
    for nid in neighbor_ids:
        assert env.agents[nid].opinion_score > 0.0
