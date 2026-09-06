# Task: interval set operations

Create a single file `interval_ops.py` exposing three functions.

Intervals are `(start, end)` pairs of integers, **half-open**: `start` is included,
`end` is excluded. An interval with `start >= end` is empty.

Every function takes and returns a *list* of intervals, and every returned list must
be **normalized**: sorted by `start`, non-overlapping, non-touching, and containing no
empty intervals. Inputs may be unsorted, overlapping, or contain empty intervals.

## `merge(intervals) -> list[tuple[int, int]]`

Normalize a list of intervals. Intervals that overlap *or touch* are combined:
`(1, 3)` and `(3, 5)` become `(1, 5)`.

## `intersect(a, b) -> list[tuple[int, int]]`

The set of points present in both `a` and `b`.

## `subtract(a, b) -> list[tuple[int, int]]`

The set of points present in `a` but not in `b`.

## Examples

```
merge([(5, 7), (1, 3), (2, 4)])            -> [(1, 4), (5, 7)]
merge([(1, 3), (3, 5)])                    -> [(1, 5)]
intersect([(1, 10)], [(2, 4), (6, 8)])     -> [(2, 4), (6, 8)]
subtract([(1, 10)], [(3, 5)])              -> [(1, 3), (5, 10)]
```

## Constraints

- Standard library only.
- Return lists of tuples, not lists of lists.
- The file must be importable as `import interval_ops` from the working directory.
