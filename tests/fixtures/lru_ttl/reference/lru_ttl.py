"""Reference solution for lru_ttl."""

import time
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity, ttl=None, clock=None):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock or time.monotonic
        self._entries = OrderedDict()  # key -> (value, inserted_at)
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def _expired(self, inserted_at, now):
        return self.ttl is not None and now - inserted_at >= self.ttl

    def _purge(self, now):
        stale = [
            key
            for key, (_, inserted_at) in self._entries.items()
            if self._expired(inserted_at, now)
        ]
        for key in stale:
            del self._entries[key]
            self._expirations += 1

    def get(self, key, default=None):
        now = self.clock()
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return default

        value, inserted_at = entry
        if self._expired(inserted_at, now):
            del self._entries[key]
            self._expirations += 1
            self._misses += 1
            return default

        self._entries.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key, value):
        now = self.clock()
        self._purge(now)

        if key in self._entries:
            del self._entries[key]
        elif len(self._entries) >= self.capacity:
            self._entries.popitem(last=False)
            self._evictions += 1

        self._entries[key] = (value, now)

    def __len__(self):
        self._purge(self.clock())
        return len(self._entries)

    def stats(self):
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "expirations": self._expirations,
        }
