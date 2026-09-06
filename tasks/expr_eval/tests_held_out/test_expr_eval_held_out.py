import pytest
from expr_eval import evaluate


@pytest.mark.parametrize(
    "a,b",
    [(7, 2), (-7, 2), (7, -2), (-7, -2), (13, 5), (-13, 5)],
)
def test_division_modulo_identity_holds(a, b):
    quotient = evaluate(f"({a}) / ({b})")
    remainder = evaluate(f"({a}) % ({b})")
    assert quotient * b + remainder == a


def test_negative_exponent_raises_value_error():
    with pytest.raises(ValueError):
        evaluate("2 ** -1")


def test_modulo_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        evaluate("5 % 0")


def test_deeply_nested_parentheses():
    assert evaluate("((((1 + 2)) * ((3))))") == 9


def test_empty_expression_raises_value_error():
    with pytest.raises(ValueError):
        evaluate("   ")


def test_unbalanced_parenthesis_raises_value_error():
    with pytest.raises(ValueError):
        evaluate("(1 + 2")


def test_unexpected_character_raises_value_error():
    with pytest.raises(ValueError):
        evaluate("1 $ 2")
