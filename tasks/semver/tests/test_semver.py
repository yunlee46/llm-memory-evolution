import pytest
from semver import compare, parse


def test_parse_plain_version():
    assert parse("1.2.3") == {
        "major": 1,
        "minor": 2,
        "patch": 3,
        "prerelease": None,
        "build": None,
    }


def test_parse_with_prerelease_and_build():
    assert parse("1.0.0-alpha.1+build.7") == {
        "major": 1,
        "minor": 0,
        "patch": 0,
        "prerelease": "alpha.1",
        "build": "build.7",
    }


def test_parse_zero_is_allowed_but_leading_zero_is_not():
    assert parse("0.0.0")["major"] == 0
    with pytest.raises(ValueError):
        parse("01.0.0")


def test_parse_rejects_missing_component():
    with pytest.raises(ValueError):
        parse("1.2")


def test_parse_rejects_empty_prerelease():
    with pytest.raises(ValueError):
        parse("1.0.0-")


def test_compare_by_major_minor_patch():
    assert compare("1.0.0", "2.0.0") == -1
    assert compare("2.1.0", "2.0.9") == 1
    assert compare("1.2.3", "1.2.3") == 0


def test_prerelease_sorts_before_release():
    assert compare("1.0.0-alpha", "1.0.0") == -1
    assert compare("1.0.0", "1.0.0-alpha") == 1


def test_numeric_prerelease_identifiers_compare_numerically():
    assert compare("1.0.0-2", "1.0.0-10") == -1


def test_numeric_identifier_sorts_before_alphanumeric():
    assert compare("1.0.0-1", "1.0.0-alpha") == -1


def test_more_identifiers_sorts_after():
    assert compare("1.0.0-alpha", "1.0.0-alpha.1") == -1


def test_build_metadata_is_ignored_for_precedence():
    assert compare("1.0.0+a", "1.0.0+b") == 0
