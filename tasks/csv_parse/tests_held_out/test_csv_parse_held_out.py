from csv_parse import parse


def test_blank_line_in_the_middle_is_one_empty_field():
    assert parse("a\n\nb") == [["a"], [""], ["b"]]


def test_two_trailing_newlines_produce_a_blank_record():
    assert parse("a,b\n\n") == [["a", "b"], [""]]


def test_empty_quoted_field():
    assert parse('a,"",b') == [["a", "", "b"]]


def test_quoted_field_containing_only_escaped_quotes():
    assert parse('""""') == [['"']]


def test_quoted_field_may_contain_crlf():
    assert parse('"a\r\nb"') == [["a\r\nb"]]


def test_single_field_no_separators():
    assert parse("solo") == [["solo"]]


def test_trailing_comma_yields_a_final_empty_field():
    assert parse("a,b,") == [["a", "b", ""]]
