"""Tests for evaluate.py -- variance, convergence, and clustering logic
against small hand-constructed logs."""

from evaluate import count_opinion_clusters, evaluate, find_convergence_round, variance_per_round


def _make_log(rounds_scores: dict[int, list[float]]) -> list[dict]:
    log = []
    for round_num, scores in rounds_scores.items():
        for agent_id, score in enumerate(scores):
            log.append({"round": round_num, "agent_id": agent_id, "opinion_score": score})
    return log


def test_variance_per_round_zero_when_all_equal():
    log = _make_log({0: [0.5, 0.5, 0.5]})
    variances = variance_per_round(log)
    assert variances[0] == 0.0


def test_variance_per_round_matches_hand_computation():
    log = _make_log({0: [0.0, 1.0]})  # mean=0.5, var = ((0-0.5)^2+(1-0.5)^2)/2 = 0.25
    variances = variance_per_round(log)
    assert variances[0] == 0.25


def test_find_convergence_round_detects_first_stable_drop():
    variances = {0: 0.5, 1: 0.2, 2: 0.005, 3: 0.02, 4: 0.001, 5: 0.0009}
    # round 2 dips below 0.01 but pops back up at round 3, so true
    # convergence (stays below threshold through the end) is round 4.
    assert find_convergence_round(variances, threshold=0.01) == 4


def test_find_convergence_round_none_if_never_converges():
    variances = {0: 0.5, 1: 0.4, 2: 0.3}
    assert find_convergence_round(variances, threshold=0.01) is None


def test_count_opinion_clusters_single_cluster_when_all_close():
    scores = [0.1, 0.12, 0.09, 0.11]
    assert count_opinion_clusters(scores, eps=0.15) == 1


def test_count_opinion_clusters_two_clusters_when_split():
    scores = [-0.9, -0.85, 0.9, 0.85]
    assert count_opinion_clusters(scores, eps=0.15) == 2


def test_evaluate_end_to_end_returns_expected_keys():
    log = _make_log({0: [0.0, 1.0], 1: [0.4, 0.6]})
    seed_info = {"keywords": ["remote", "work"], "bias": 0.2}
    metrics = evaluate("remote work", seed_info, log, threshold=0.01, cluster_eps=0.15)

    assert set(metrics) >= {
        "variance_per_round", "convergence_round", "num_clusters",
        "initial_variance", "final_variance", "threshold", "summary",
    }
    assert isinstance(metrics["summary"], str)
    assert "remote work" in metrics["summary"]
