# Task: integer expression evaluator

Create a single file `expr_eval.py` exposing `evaluate(expression: str) -> int`.

It evaluates an arithmetic expression over integers.

## Grammar

- Integer literals: one or more digits. A literal never carries its own sign.
- Binary operators: `+`, `-`, `*`, `/`, `%`, `**`
- Unary `+` and `-`
- Parentheses

## Precedence, tightest first

1. `**` — right associative, so `2 ** 3 ** 2` is `512`
2. unary `+` / `-`
3. `*`, `/`, `%` — left associative
4. `+`, `-` — left associative

Unary minus binds *looser* than `**`, so `-2 ** 2` is `-4`, not `4`.

## Semantics

- `/` is integer division truncating **toward zero**: `-7 / 2` is `-3`, not `-4`.
- `%` takes the sign of the **left** operand: `-7 % 3` is `-1`, and `7 % -3` is `1`.
  The identity `(a / b) * b + (a % b) == a` must hold for all valid `a`, `b`.
- `**` with a negative exponent raises `ValueError`.
- Division or modulo by zero raises `ZeroDivisionError`.
- Whitespace is insignificant.

## Errors

Raise `ValueError` for any malformed expression: unexpected characters,
unbalanced parentheses, a missing operand, or an empty expression.

## Constraints

- Standard library only. Do **not** use `eval`, `exec`, `ast`, or `compile`.
- The file must be importable as `import expr_eval` from the working directory.
