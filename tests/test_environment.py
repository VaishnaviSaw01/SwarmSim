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


def test_persona_includes_influence_trait_in_range():
    env = Environment(num_agents=20, k=4, rewire_prob=0.1, seed=1)
    for agent in env.agents.values():
        assert 0.05 <= agent.persona["influence"] <= 1.0


def test_graph_data_reports_nodes_and_edges():
    env = Environment(num_agents=10, k=4, rewire_prob=0.1, seed=1)
    data = env.graph_data()

    assert len(data["nodes"]) == 10
    assert all({"id", "stubbornness", "openness", "influence"} <= set(n) for n in data["nodes"])
    assert len(data["edges"]) == env.graph.number_of_edges()
    assert all(len(e) == 2 for e in data["edges"])


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


def test_step_weights_neighbor_contribution_by_influence():
    """A high-influence neighbor should pull an agent's opinion harder
    than a low-influence one -- the point of the DeGroot-weighted rule."""
    env = Environment(num_agents=10, k=2, rewire_prob=0.0, seed=2)
    target = 0
    neighbor_ids = list(env.graph.neighbors(target))
    assert len(neighbor_ids) >= 2, "test needs at least 2 neighbors to compare weights"

    for agent in env.agents.values():
        agent.persona["stubbornness"] = 0.0  # fully swayed by neighbors
        agent.opinion_score = 0.0
    env.agents[target].persona["stubbornness"] = 0.0

    high_influence_id, low_influence_id = neighbor_ids[0], neighbor_ids[1]
    env.agents[high_influence_id].persona["influence"] = 1.0
    env.agents[high_influence_id].opinion_score = 1.0
    env.agents[low_influence_id].persona["influence"] = 0.05
    env.agents[low_influence_id].opinion_score = -1.0
    for nid in neighbor_ids[2:]:
        env.agents[nid].opinion_score = 0.0

    env.step(round_num=1, noise_std=0.0)

    # The high-influence neighbor (opinion +1.0) should pull target's
    # opinion positive overall, even though a low-influence neighbor at
    # -1.0 (and other neutral neighbors) are also in the mix -- a plain
    # unweighted mean would pull it toward 0, not clearly positive.
    assert env.agents[target].opinion_score > 0.05


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
