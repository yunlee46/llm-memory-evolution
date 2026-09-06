import pytest
from lru_ttl import LRUCache


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


def test_put_then_get():
    cache = LRUCache(2)
    cache.put("a", 1)
    assert cache.get("a") == 1


def test_missing_key_returns_default():
    cache = LRUCache(2)
    assert cache.get("nope") is None
    assert cache.get("nope", "fallback") == "fallback"


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        LRUCache(0)


def test_least_recently_used_is_evicted():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_get_refreshes_recency():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_updating_a_key_refreshes_recency():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 99)
    cache.put("c", 3)
    assert cache.get("a") == 99
    assert cache.get("b") is None


def test_entry_expires_after_ttl(clock):
    cache = LRUCache(2, ttl=10, clock=clock)
    cache.put("a", 1)
    clock.advance(10)
    assert cache.get("a") is None


def test_entry_is_live_just_before_ttl(clock):
    cache = LRUCache(2, ttl=10, clock=clock)
    cache.put("a", 1)
    clock.advance(9.9)
    assert cache.get("a") == 1


def test_len_counts_only_live_entries(clock):
    cache = LRUCache(4, ttl=10, clock=clock)
    cache.put("a", 1)
    clock.advance(5)
    cache.put("b", 2)
    clock.advance(6)
    assert len(cache) == 1


def test_stats_shape_and_counts():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.get("a")
    cache.get("missing")
    assert cache.stats() == {
        "hits": 1,
        "misses": 1,
        "evictions": 0,
        "expirations": 0,
    }


def test_eviction_is_counted():
    cache = LRUCache(1)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.stats()["evictions"] == 1
