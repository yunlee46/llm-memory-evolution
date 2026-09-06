from glob_match import match


def test_double_star_may_match_nothing():
    assert match("**main.py", "main.py")


def test_double_star_does_not_imply_an_optional_separator():
    """`**` is defined as characters, not path segments — the `/` is required."""
    assert not match("**/main.py", "main.py")


def test_leading_bracket_literal_close_bracket():
    assert match("[]]", "]")


def test_negated_class_with_literal_close_bracket():
    assert not match("[!]]", "]")


def test_dash_at_class_edges_is_literal():
    assert match("[-a]", "-")
    assert match("[a-]", "-")


def test_character_class_never_matches_separator():
    assert not match("[!a]", "/")


def test_backtracking_star_does_not_stop_at_first_opportunity():
    assert match("*.txt", "a.txt.txt")


def test_many_stars_do_not_blow_up():
    assert not match("*a*a*a*a*a*a*b", "a" * 40)
