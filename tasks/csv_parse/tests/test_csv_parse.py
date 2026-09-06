import pytest
from csv_parse import parse


def test_simple_grid():
    assert parse("a,b\nc,d") == [["a", "b"], ["c", "d"]]


def test_quoted_field_may_contain_a_comma():
    assert parse('"a,b",c') == [["a,b", "c"]]


def test_doubled_quote_is_an_escaped_quote():
    assert parse('"say ""hi""",x') == [['say "hi"', "x"]]


def test_empty_fields_are_preserved():
    assert parse("a,,b") == [["a", "", "b"]]


def test_empty_input_is_no_records():
    assert parse("") == []


def test_crlf_separates_records():
    assert parse("a,b\r\nc,d") == [["a", "b"], ["c", "d"]]


def test_single_trailing_newline_adds_no_record():
    assert parse("a,b\n") == [["a", "b"]]


def test_quoted_field_may_contain_a_newline():
    assert parse('"line1\nline2",x') == [["line1\nline2", "x"]]


def test_unquoted_whitespace_is_not_stripped():
    assert parse(" a , b ") == [[" a ", " b "]]


def test_unterminated_quote_raises():
    with pytest.raises(ValueError):
        parse('"abc')


def test_text_after_closing_quote_raises():
    with pytest.raises(ValueError):
        parse('"abc"x')
