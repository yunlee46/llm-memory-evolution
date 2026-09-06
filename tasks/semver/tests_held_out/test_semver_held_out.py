import pytest
from semver import compare, parse


def test_full_precedence_chain_from_the_spec():
    ordered = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
    ]
    for earlier, later in zip(ordered, ordered[1:]):
        assert compare(earlier, later) == -1, f"{earlier} should precede {later}"
        assert compare(later, earlier) == 1


def test_build_metadata_permits_leading_zeros():
    assert parse("1.0.0+007")["build"] == "007"


def test_numeric_prerelease_identifier_rejects_leading_zeros():
    with pytest.raises(ValueError):
        parse("1.0.0-01")


def test_hyphen_is_a_valid_identifier_character():
    assert parse("1.0.0-alpha-9")["prerelease"] == "alpha-9"


def test_empty_build_identifier_is_rejected():
    with pytest.raises(ValueError):
        parse("1.0.0+a..b")


def test_prerelease_identifier_rejects_invalid_characters():
    with pytest.raises(ValueError):
        parse("1.0.0-al_pha")


def test_compare_is_antisymmetric_across_build_metadata():
    assert compare("1.0.0-alpha+x", "1.0.0-alpha+y") == 0
