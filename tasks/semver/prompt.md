# Task: semantic version parsing and precedence

Create a single file `semver.py` exposing two functions.

## `parse(version: str) -> dict`

Parse a semantic version string into a dict with exactly these keys:

```python
{"major": int, "minor": int, "patch": int, "prerelease": str | None, "build": str | None}
```

The format is `MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]`.

- `MAJOR`, `MINOR`, `PATCH` are non-negative integers with no leading zeros
  (`0` itself is fine, `01` is not).
- `PRERELEASE` is a dot-separated series of identifiers. Each identifier is
  non-empty and contains only `[0-9A-Za-z-]`. A purely numeric identifier must
  not have leading zeros.
- `BUILD` is a dot-separated series of non-empty `[0-9A-Za-z-]` identifiers.
  Leading zeros are permitted here.
- Store `prerelease` and `build` as the raw string between their delimiters, or
  `None` when absent.

Raise `ValueError` on anything that does not conform.

## `compare(a: str, b: str) -> int`

Return `-1` if `a` sorts before `b`, `1` if after, `0` if they have equal
precedence. Precedence rules:

1. Compare `major`, then `minor`, then `patch` numerically.
2. A version *with* a prerelease sorts **before** the same version without one.
   So `1.0.0-alpha` < `1.0.0`.
3. Otherwise compare prerelease identifiers left to right:
   - numeric identifiers compare numerically,
   - a numeric identifier always sorts before a non-numeric one,
   - non-numeric identifiers compare by ASCII order,
   - if all shared identifiers are equal, the version with **more** identifiers
     sorts after. So `1.0.0-alpha` < `1.0.0-alpha.1`.
4. `BUILD` is ignored entirely for precedence. `1.0.0+a` and `1.0.0+b` are equal.

## Constraints

- Standard library only.
- The file must be importable as `import semver` from the working directory.
