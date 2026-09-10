"""Tests for evaluate.py -- variance, convergence, and clustering logic
against small hand-constructed logs."""

from evaluate import (
    classify_outcome,
    cluster_breakdown,
    count_opinion_clusters,
    evaluate,
    find_convergence_round,
    opinion_distribution,
    swing_agents,
    variance_per_round,
)


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


def test_opinion_distribution_buckets_by_the_shared_lean_threshold():
    scores = [0.1, 0.12, 0.09, 0.11]  # all within +-0.15 of zero -> neutral
    dist = opinion_distribution(scores)
    assert dist["neutral"] == {"count": 4, "pct": 100.0}
    assert dist["favorable"]["count"] == 0
    assert dist["opposed"]["count"] == 0


def test_opinion_distribution_splits_favorable_and_opposed():
    scores = [0.9, 0.85, -0.9, -0.2]  # 2 favorable, 2 opposed
    dist = opinion_distribution(scores)
    assert dist["favorable"] == {"count": 2, "pct": 50.0}
    assert dist["opposed"] == {"count": 2, "pct": 50.0}
    assert dist["neutral"] == {"count": 0, "pct": 0.0}


def test_cluster_breakdown_reports_size_mean_and_lean_per_cluster():
    scores = [-0.9, -0.85, 0.9, 0.85]
    clusters = cluster_breakdown(scores, eps=0.15)

    assert len(clusters) == 2
    assert {c["lean"] for c in clusters} == {"favorable", "opposed"}
    for c in clusters:
        assert c["size"] == 2
        assert c["pct"] == 50.0


def test_classify_outcome_labels_single_cluster_as_consensus():
    clusters = cluster_breakdown([0.1, 0.12, 0.09, 0.11], eps=0.15)
    assert classify_outcome(clusters) == "Consensus"


def test_classify_outcome_labels_two_even_clusters_as_polarized():
    clusters = cluster_breakdown([-0.9, -0.85, 0.9, 0.85], eps=0.15)
    assert classify_outcome(clusters) == "Polarized"


def test_classify_outcome_labels_dominant_plus_holdouts():
    # 8 agents near 0.5 (one dominant cluster), 1 agent way off at -0.9.
    scores = [0.5] * 8 + [-0.9]
    clusters = cluster_breakdown(scores, eps=0.15)
    assert classify_outcome(clusters) == "Majority with holdouts"


def test_swing_agents_finds_biggest_mover_and_most_steadfast():
    log = _make_log({0: [0.0, 0.5, -0.5], 3: [0.1, 0.9, -0.55]})
    swing = swing_agents(log)

    assert swing["most_persuaded"]["agent_id"] == 1  # 0.5 -> 0.9, moved 0.4
    assert swing["most_persuaded"]["movement"] == 0.4
    assert swing["most_steadfast"]["agent_id"] == 2  # -0.5 -> -0.55, moved -0.05


def test_swing_agents_none_for_empty_log():
    assert swing_agents([]) is None


def test_evaluate_end_to_end_returns_expected_keys():
    log = _make_log({0: [0.0, 1.0], 1: [0.4, 0.6]})
    seed_info = {"keywords": ["remote", "work"], "bias": 0.2}
    metrics = evaluate("remote work", seed_info, log, threshold=0.01, cluster_eps=0.15)

    assert set(metrics) >= {
        "variance_per_round", "convergence_round", "num_clusters", "clusters",
        "distribution", "verdict", "swing",
        "initial_variance", "final_variance", "threshold", "summary",
    }
    assert isinstance(metrics["summary"], str)
    assert "remote work" in metrics["summary"]
    assert metrics["verdict"] in {"Consensus", "Polarized", "Majority with holdouts", "No data"}
    assert len(metrics["clusters"]) == metrics["num_clusters"]
