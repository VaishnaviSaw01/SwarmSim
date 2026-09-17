"""Tests for seed.py -- TF-IDF keyword extraction and lexicon-based bias.

Includes regression coverage for the "why is everything neutral" issue:
compute_bias must return exactly 0.0 for topics with no lexicon hits (a
real, documented limitation of a fixed word list -- see seed.py's module
docstring), and a clearly nonzero bias once the topic actually contains
words from the (now broader) lexicon.
"""

import pytest

from seed import compute_bias, extract_keywords, seed_from_topic


def test_extract_keywords_respects_top_n():
    keywords = extract_keywords("new company policy on remote work", top_n=3)
    assert len(keywords) == 3


def test_extract_keywords_drops_stopwords():
    keywords = extract_keywords("the new parking policy")
    assert "the" not in keywords
    assert "policy" in keywords


def test_extract_keywords_empty_for_stopwords_only():
    assert extract_keywords("the a an of") == []


def test_compute_bias_zero_for_empty_keywords():
    assert compute_bias([]) == 0.0


def test_compute_bias_zero_when_no_lexicon_hits():
    # None of these are sentiment-bearing words -- 0.0 is the correct,
    # honest answer here, not a bug (see seed.py's module docstring).
    assert compute_bias(["company", "new", "policy", "remote", "work"]) == 0.0


def test_compute_bias_all_positive():
    assert compute_bias(["excellent", "helpful", "safe"]) == pytest.approx(1.0)


def test_compute_bias_all_negative():
    assert compute_bias(["toxic", "hostile", "unsafe"]) == pytest.approx(-1.0)


def test_compute_bias_mixed_nets_out():
    # 2 positive, 1 negative, 2 neutral -> (2 - 1) / 5 = 0.2
    keywords = ["excellent", "helpful", "toxic", "quarterly", "review"]
    assert compute_bias(keywords) == pytest.approx(0.2)


def test_broader_lexicon_catches_general_sentiment_not_just_workplace_words():
    """Regression test: the original ~30-word-per-list lexicon was heavily
    workplace-flavored and missed everyday positive/negative phrasing
    entirely. These topics have no workplace-specific charged words at
    all, but do contain general-sentiment adjectives the expanded lists
    added."""
    pos_keywords = extract_keywords("this new safety regulation is excellent for workers")
    assert compute_bias(pos_keywords) > 0

    neg_keywords = extract_keywords("the merger creates a toxic and hostile work environment")
    assert compute_bias(neg_keywords) < 0


def test_seed_from_topic_wires_keywords_and_bias_together():
    keywords, bias = seed_from_topic("mandatory return to office is a burden and unfair surveillance")
    assert "mandatory" in keywords
    assert bias < 0
