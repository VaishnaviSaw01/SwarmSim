"""Tests for agent.py.

The update rule is the one piece of math in this project that must be
exactly right, so most of these tests check the formula against
hand-computed expected values rather than just "does it run".
"""

from agent import Agent


def test_opinion_clipped_on_init():
    a = Agent(agent_id=0, persona={"stubbornness": 0.5}, opinion_score=5.0)
    assert a.opinion_score == 1.0

    b = Agent(agent_id=1, persona={"stubbornness": 0.5}, opinion_score=-5.0)
    assert b.opinion_score == -1.0


def test_update_opinion_matches_formula():
    a = Agent(agent_id=0, persona={"stubbornness": 0.8}, opinion_score=0.2)
    neighbor_scores = [0.6, 1.0]  # mean = 0.8
    expected = 0.8 * 0.2 + 0.2 * 0.8 + 0.0  # w*old + (1-w)*avg + noise(=0)

    result = a.update_opinion(neighbor_scores, noise=0.0, round_num=1)

    assert result == expected
    assert a.opinion_score == expected


def test_update_opinion_no_neighbors_keeps_own_score_plus_noise():
    a = Agent(agent_id=0, persona={"stubbornness": 0.5}, opinion_score=0.3)
    result = a.update_opinion([], noise=0.1, round_num=1)
    assert result == 0.4  # w*0.3 + (1-w)*0.3 (falls back to own score) + 0.1


def test_update_opinion_result_stays_clipped():
    a = Agent(agent_id=0, persona={"stubbornness": 0.0}, opinion_score=0.0)
    result = a.update_opinion([1.0], noise=5.0, round_num=1)
    assert result == 1.0


def test_memory_caps_at_five_entries():
    a = Agent(agent_id=0, persona={"stubbornness": 0.5})
    for r in range(10):
        a.update_opinion([0.1], noise=0.0, round_num=r)
    assert len(a.memory) == 5
    assert a.memory[-1]["round"] == 9


def test_to_dict_contains_expected_keys():
    a = Agent(agent_id=3, persona={"stubbornness": 0.4, "openness": 0.7}, opinion_score=0.25)
    d = a.to_dict()
    assert d == {
        "agent_id": 3,
        "persona": {"stubbornness": 0.4, "openness": 0.7},
        "opinion_score": 0.25,
    }
