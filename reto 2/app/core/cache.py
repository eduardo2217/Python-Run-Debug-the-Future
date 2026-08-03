"""Thread-safe TTL cache used to avoid redundant calls to the AI provider.

Bonus B — caching: identical requests are served from an in-memory cache for a
configurable TTL, so repeated inputs never hit the external API again.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class _Entry:
    value: V
    expires_at: float


@dataclass
class TTLCache(Generic[K, V]):
    """Simple, thread-safe, size-bounded cache with time-based expiry.

    Uses an LRU ``OrderedDict`` for O(1) lookups, a lock for thread safety,
    and lazily evicts expired and overflow entries on access.
    """

    ttl_seconds: float = 3600.0
    max_size: int = field(default=1024)

    def __post_init__(self) -> None:
        self._dict: OrderedDict[K, _Entry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        """Return the cached value for ``key`` or ``None`` if missing/expired."""
        with self._lock:
            entry = self._dict.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._dict[key]
                return None
            # Refresh recency (LRU ordering) on a hit.
            self._dict.move_to_end(key)
            return entry.value

    def set(self, key: K, value: V) -> None:
        """Store ``value`` under ``key`` with an absolute expiration."""
        if self.ttl_seconds <= 0:
            return
        with self._lock:
            self._dict[key] = _Entry(
                value=value, expires_at=time.monotonic() + self.ttl_seconds
            )
            self._dict.move_to_end(key)
            while len(self._dict) > self.max_size:
                self._dict.popitem(last=False)

    def invalidate(self, key: K) -> bool:
        """Remove ``key`` from the cache; returns whether it existed."""
        with self._lock:
            return self._dict.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._dict.clear()

    def __len__(self) -> int:
        """Number of currently stored (unexpired) entries."""
        with self._lock:
            return len([k for k in self._dict.values() if k.expires_at > time.monotonic()])