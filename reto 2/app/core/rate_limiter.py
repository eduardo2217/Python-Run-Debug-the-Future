"""Sliding-window rate limiter for the API (Bonus B).

Protects the endpoint from abuse and from exhausting the upstream provider's
quota by bounding how many requests each client can make within a window.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class SlidingWindowRateLimiter:
    """Per-client sliding-window limiter.

    For each unique client key (e.g. client IP) it keeps a deque of request
    timestamps. A request is allowed when the number of timestamps inside the
    window is below ``max_requests``.
    """

    max_requests: int = 10
    window_seconds: float = 60.0
    _buckets: dict = field(default_factory=lambda: defaultdict(deque))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, key: str) -> bool:
        """Register (if allowed) and return whether the request is admitted.

        A request consumes a token only when admitted, preventing a flood of
        rejected requests from starving out subsequent legitimate calls.
        """
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets[key]
            # Drop timestamps that have left the sliding window.
            cutoff = now - self.window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True

    def _purge(self, key: str) -> None:
        """Remove all recorded timestamps for a key (useful in tests/admin)."""
        with self._lock:
            self._buckets.pop(key, None)