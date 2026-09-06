# Task: LRU cache with TTL

Create a single file `lru_ttl.py` exposing a class `LRUCache`.

```python
LRUCache(capacity: int, ttl: float | None = None, clock=None)
```

- `capacity` — maximum number of live entries. Must be a positive integer;
  otherwise raise `ValueError`.
- `ttl` — seconds an entry stays valid. `None` means entries never expire.
- `clock` — a zero-argument callable returning the current time as a float.
  Defaults to `time.monotonic`. **Only** call the clock through this attribute;
  the tests inject a fake one.

## Methods

### `put(key, value) -> None`

Insert or update. Updating an existing key refreshes both its value and its
recency, and restarts its TTL. When inserting would exceed `capacity`, evict the
least recently used live entry first.

### `get(key, default=None)`

Return the value, or `default` if the key is absent or expired. A successful
`get` refreshes recency but does **not** restart the TTL.

### `__len__()`

The number of live (non-expired) entries. Reading the length must not itself
count as a use of any entry.

### `stats() -> dict`

Return exactly `{"hits": int, "misses": int, "evictions": int, "expirations": int}`.

- `hits` — `get` calls that returned a live value.
- `misses` — `get` calls that returned the default, for any reason.
- `evictions` — entries removed to stay within `capacity`.
- `expirations` — entries removed because their TTL elapsed.

An expired entry that is then read counts as **one** miss and **one** expiration.

## Expiry semantics

An entry inserted at time `t` is live while `now - t < ttl`, and expired once
`now - t >= ttl`. Expired entries must never be returned, must not count toward
`__len__`, and must not be chosen as the LRU eviction victim in preference to a
live one — drop them instead.

## Constraints

- Standard library only. Do **not** use `functools.lru_cache`.
- The file must be importable as `import lru_ttl` from the working directory.
