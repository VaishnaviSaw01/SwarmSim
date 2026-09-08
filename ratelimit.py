"""
ratelimit.py -- a minimal in-memory, per-client sliding-window rate limiter.

WHY hand-rolled instead of a library: this project's whole premise is that
every piece is simple enough to explain from memory. A sliding window
counter kept in a plain dict answers exactly one question -- "how many
requests has this client made in the last N seconds, and is that over the
limit" -- with no dependency and nothing hidden inside a decorator from a
third-party package.

WHY this matters for /simulate specifically: unlike a normal CRUD
endpoint, /simulate does real CPU work (up to ~1s per request at the
service's parameter caps). Publicly exposed with no limiter, a handful of
concurrent large requests would be enough to starve the single process
this app runs as. The limiter is deliberately generous (it exists to stop
abuse, not to meter normal use) and sits in front of the expensive
endpoint only.

WHY in-memory, not Redis/a database: this is meant to run as a single
process (one Render/uvicorn worker, as configured in render.yaml). State
resets on restart and is not shared across multiple workers or instances
-- a documented limitation, not an oversight. A multi-worker deployment
would need to move this to a shared store to stay correct; a single-
process portfolio deployment does not.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    """Tracks recent request timestamps per client key (e.g. IP address).

    Args:
        max_requests: how many requests a single key may make within
            window_seconds before being refused.
        window_seconds: length of the sliding window, in seconds.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, float]:
        """Record an attempt for `key` and say whether it's allowed.

        WHY a deque of timestamps rather than a simple counter: a fixed
        window counter ("N requests since the top of this minute") lets a
        client burst up to 2x the limit right across a window boundary. A
        sliding window -- drop timestamps older than window_seconds, then
        check how many remain -- doesn't have that edge case, and is still
        just arithmetic over a list of floats.

        Args:
            key: identifies the client (typically their IP address).

        Returns:
            (allowed, retry_after_seconds). When allowed is False,
            retry_after_seconds is how long the client should wait before
            trying again (used to set the Retry-After response header).
        """
        now = time.monotonic()
        hits = self._hits[key]

        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()

        if len(hits) >= self.max_requests:
            retry_after = self.window_seconds - (now - hits[0])
            return False, max(retry_after, 0.0)

        hits.append(now)
        return True, 0.0
