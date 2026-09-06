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


def test_expired_read_counts_one_miss_and_one_expiration(clock):
    cache = LRUCache(2, ttl=10, clock=clock)
    cache.put("a", 1)
    clock.advance(10)
    cache.get("a")
    stats = cache.stats()
    assert stats["misses"] == 1
    assert stats["expirations"] == 1
    assert stats["hits"] == 0


def test_get_does_not_restart_the_ttl(clock):
    cache = LRUCache(2, ttl=10, clock=clock)
    cache.put("a", 1)
    clock.advance(6)
    assert cache.get("a") == 1
    clock.advance(5)
    assert cache.get("a") is None


def test_put_restarts_the_ttl(clock):
    cache = LRUCache(2, ttl=10, clock=clock)
    cache.put("a", 1)
    clock.advance(6)
    cache.put("a", 2)
    clock.advance(6)
    assert cache.get("a") == 2


def test_expired_entry_is_dropped_before_a_live_one_is_evicted(clock):
    cache = LRUCache(2, ttl=10, clock=clock)
    cache.put("stale", 1)
    clock.advance(5)
    cache.put("fresh", 2)
    clock.advance(6)
    cache.put("new", 3)
    assert cache.get("fresh") == 2
    assert cache.get("new") == 3
    assert cache.stats()["evictions"] == 0


def test_len_does_not_count_as_a_use(clock):
    cache = LRUCache(2, clock=clock)
    cache.put("a", 1)
    cache.put("b", 2)
    len(cache)
    cache.put("c", 3)
    assert cache.get("a") is None


def test_ttl_none_never_expires(clock):
    cache = LRUCache(2, ttl=None, clock=clock)
    cache.put("a", 1)
    clock.advance(10_000)
    assert cache.get("a") == 1


def test_none_is_a_storable_value():
    cache = LRUCache(2)
    cache.put("a", None)
    assert cache.get("a", "default") is None
    assert cache.stats()["hits"] == 1
