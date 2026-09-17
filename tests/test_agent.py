"""Tests for agent.py.

The update rule is the one piece of math in this project that must be
exactly right, so most of these tests check the formula against
hand-computed expected values rather than just "does it run".
"""

import pytest

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

    # pytest.approx, not ==: this formula is plain float arithmetic, so it
    # is subject to ordinary floating-point rounding (e.g. 0.32 vs.
    # 0.32000000000000006) -- the rule is still exact, the bit pattern isn't.
    assert result == pytest.approx(expected)
    assert a.opinion_score == pytest.approx(expected)


def test_update_opinion_no_neighbors_keeps_own_score_plus_noise():
    a = Agent(agent_id=0, persona={"stubbornness": 0.5}, opinion_score=0.3)
    result = a.update_opinion([], noise=0.1, round_num=1)
    assert result == pytest.approx(0.4)  # w*0.3 + (1-w)*0.3 (falls back to own score) + 0.1


def test_update_opinion_result_stays_clipped():
    a = Agent(agent_id=0, persona={"stubbornness": 0.0}, opinion_score=0.0)
    result = a.update_opinion([1.0], noise=5.0, round_num=1)
    assert result == 1.0


def test_update_opinion_weights_neighbors_by_influence():
    a = Agent(agent_id=0, persona={"stubbornness": 0.0}, opinion_score=0.0)
    # Neighbor 0 says 1.0 with weight 3 (high influence); neighbor 1 says
    # -1.0 with weight 1 (low influence). Weighted mean = (1*3 + -1*1) / 4 = 0.5,
    # not the plain mean of 0.0 -- this is exactly the case a plain
    # average can't distinguish but a DeGroot-weighted one can.
    result = a.update_opinion([1.0, -1.0], noise=0.0, round_num=1, neighbor_weights=[3.0, 1.0])
    assert result == pytest.approx(0.5)


def test_update_opinion_equal_weights_matches_plain_mean():
    a = Agent(agent_id=0, persona={"stubbornness": 0.5}, opinion_score=0.2)
    unweighted = Agent(agent_id=1, persona={"stubbornness": 0.5}, opinion_score=0.2)

    weighted_result = a.update_opinion([0.6, 1.0], noise=0.0, round_num=1, neighbor_weights=[1.0, 1.0])
    plain_result = unweighted.update_opinion([0.6, 1.0], noise=0.0, round_num=1)

    assert weighted_result == pytest.approx(plain_result)


def test_update_opinion_zero_total_weight_falls_back_to_plain_mean():
    a = Agent(agent_id=0, persona={"stubbornness": 0.0}, opinion_score=0.0)
    result = a.update_opinion([1.0, -1.0], noise=0.0, round_num=1, neighbor_weights=[0.0, 0.0])
    assert result == pytest.approx(0.0)  # plain mean of [1.0, -1.0]


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
