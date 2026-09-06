"""Reference solution for semver."""

_IDENT_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-")


def _parse_number(text):
    if not text or not text.isdigit():
        raise ValueError(f"invalid numeric component {text!r}")
    if len(text) > 1 and text[0] == "0":
        raise ValueError(f"leading zero in {text!r}")
    return int(text)


def _check_identifiers(raw, *, numeric_leading_zeros_allowed):
    identifiers = raw.split(".")
    for identifier in identifiers:
        if not identifier:
            raise ValueError("empty identifier")
        if not set(identifier) <= _IDENT_CHARS:
            raise ValueError(f"invalid character in identifier {identifier!r}")
        if (
            not numeric_leading_zeros_allowed
            and identifier.isdigit()
            and len(identifier) > 1
            and identifier[0] == "0"
        ):
            raise ValueError(f"leading zero in numeric identifier {identifier!r}")
    return identifiers


def parse(version):
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty string")

    rest = version
    build = None
    if "+" in rest:
        rest, _, build = rest.partition("+")
        _check_identifiers(build, numeric_leading_zeros_allowed=True)

    prerelease = None
    if "-" in rest:
        rest, _, prerelease = rest.partition("-")
        _check_identifiers(prerelease, numeric_leading_zeros_allowed=False)

    parts = rest.split(".")
    if len(parts) != 3:
        raise ValueError(f"expected MAJOR.MINOR.PATCH, got {rest!r}")

    return {
        "major": _parse_number(parts[0]),
        "minor": _parse_number(parts[1]),
        "patch": _parse_number(parts[2]),
        "prerelease": prerelease,
        "build": build,
    }


def _sign(value):
    return (value > 0) - (value < 0)


def _compare_identifier(left, right):
    left_numeric = left.isdigit()
    right_numeric = right.isdigit()
    if left_numeric and right_numeric:
        return _sign(int(left) - int(right))
    if left_numeric != right_numeric:
        return -1 if left_numeric else 1
    return _sign((left > right) - (left < right))


def _compare_prerelease(left, right):
    if left is None and right is None:
        return 0
    if left is None:
        return 1
    if right is None:
        return -1

    left_parts = left.split(".")
    right_parts = right.split(".")
    for left_item, right_item in zip(left_parts, right_parts):
        result = _compare_identifier(left_item, right_item)
        if result:
            return result
    return _sign(len(left_parts) - len(right_parts))


def compare(a, b):
    first = parse(a)
    second = parse(b)
    for key in ("major", "minor", "patch"):
        result = _sign(first[key] - second[key])
        if result:
            return result
    return _compare_prerelease(first["prerelease"], second["prerelease"])
