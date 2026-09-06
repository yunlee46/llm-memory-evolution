import pytest
from glob_match import match


def test_literal_match():
    assert match("main.py", "main.py")
    assert not match("main.py", "other.py")


def test_star_matches_within_a_segment():
    assert match("*.py", "main.py")


def test_star_does_not_cross_a_separator():
    assert not match("*.py", "src/main.py")


def test_double_star_crosses_separators():
    assert match("**/*.py", "src/a/main.py")


def test_question_mark_matches_one_character():
    assert match("file?.txt", "file1.txt")
    assert not match("file?.txt", "file12.txt")


def test_question_mark_does_not_match_separator():
    assert not match("a?b", "a/b")


def test_character_class_set():
    assert match("[a-c]at", "bat")
    assert not match("[a-c]at", "dat")


def test_negated_character_class():
    assert not match("[!a-c]at", "bat")
    assert match("[!a-c]at", "dat")


def test_whole_text_must_match():
    assert not match("main", "main.py")


def test_star_matches_empty_string():
    assert match("a*b", "ab")


def test_unterminated_class_raises_value_error():
    with pytest.raises(ValueError):
        match("[abc", "a")
