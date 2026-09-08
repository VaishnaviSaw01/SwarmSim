"""Tests for ratelimit.py -- the sliding-window counter that guards
POST /simulate from abuse in a public deployment."""

import time

from ratelimit import RateLimiter


def test_allows_requests_under_the_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        allowed, _ = limiter.allow("client-a")
        assert allowed is True


def test_blocks_requests_over_the_limit():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.allow("client-a")
    limiter.allow("client-a")
    allowed, retry_after = limiter.allow("client-a")
    assert allowed is False
    assert retry_after > 0


def test_clients_are_tracked_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    allowed_a, _ = limiter.allow("client-a")
    allowed_b, _ = limiter.allow("client-b")
    assert allowed_a is True
    assert allowed_b is True  # different key, independent budget


def test_old_hits_expire_out_of_the_window():
    limiter = RateLimiter(max_requests=1, window_seconds=0.05)
    allowed_first, _ = limiter.allow("client-a")
    assert allowed_first is True

    blocked, _ = limiter.allow("client-a")
    assert blocked is False

    time.sleep(0.06)  # let the window pass
    allowed_after_wait, _ = limiter.allow("client-a")
    assert allowed_after_wait is True
