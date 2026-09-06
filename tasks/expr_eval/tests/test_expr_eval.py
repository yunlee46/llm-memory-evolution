import pytest
from expr_eval import evaluate


def test_single_literal():
    assert evaluate("42") == 42


def test_addition_and_subtraction_are_left_associative():
    assert evaluate("10 - 3 - 2") == 5


def test_multiplication_binds_tighter_than_addition():
    assert evaluate("2 + 3 * 4") == 14


def test_parentheses_override_precedence():
    assert evaluate("(2 + 3) * 4") == 20


def test_unary_minus():
    assert evaluate("-5 + 2") == -3


def test_power_is_right_associative():
    assert evaluate("2 ** 3 ** 2") == 512


def test_unary_minus_binds_looser_than_power():
    assert evaluate("-2 ** 2") == -4


def test_division_truncates_toward_zero():
    assert evaluate("-7 / 2") == -3
    assert evaluate("7 / 2") == 3


def test_modulo_takes_sign_of_left_operand():
    assert evaluate("-7 % 3") == -1
    assert evaluate("7 % -3") == 1


def test_whitespace_is_insignificant():
    assert evaluate("  1+2   *  3 ") == 7


def test_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        evaluate("1 / 0")


def test_malformed_expression_raises_value_error():
    with pytest.raises(ValueError):
        evaluate("2 +")
