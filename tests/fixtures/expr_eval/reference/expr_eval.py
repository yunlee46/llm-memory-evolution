"""Reference solution for expr_eval. Validates the test suite is satisfiable."""


def _tokenize(expression):
    tokens = []
    index = 0
    length = len(expression)
    while index < length:
        char = expression[index]
        if char.isspace():
            index += 1
        elif char.isdigit():
            end = index
            while end < length and expression[end].isdigit():
                end += 1
            tokens.append(int(expression[index:end]))
            index = end
        elif expression.startswith("**", index):
            tokens.append("**")
            index += 2
        elif char in "+-*/%()":
            tokens.append(char)
            index += 1
        else:
            raise ValueError(f"unexpected character {char!r}")
    return tokens


def _div(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    magnitude = abs(a) // abs(b)
    return magnitude if (a >= 0) == (b >= 0) else -magnitude


def _mod(a, b):
    if b == 0:
        raise ZeroDivisionError("modulo by zero")
    return a - _div(a, b) * b


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self):
        token = self.peek()
        if token is None:
            raise ValueError("unexpected end of expression")
        self.pos += 1
        return token

    def expect_end(self):
        if self.pos != len(self.tokens):
            raise ValueError("trailing input")

    def expr(self):
        value = self.term()
        while self.peek() in ("+", "-"):
            operator = self.advance()
            right = self.term()
            value = value + right if operator == "+" else value - right
        return value

    def term(self):
        value = self.unary()
        while self.peek() in ("*", "/", "%"):
            operator = self.advance()
            right = self.unary()
            if operator == "*":
                value = value * right
            elif operator == "/":
                value = _div(value, right)
            else:
                value = _mod(value, right)
        return value

    def unary(self):
        token = self.peek()
        if token == "-":
            self.advance()
            return -self.unary()
        if token == "+":
            self.advance()
            return self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek() == "**":
            self.advance()
            exponent = self.unary()
            if exponent < 0:
                raise ValueError("negative exponent")
            return base**exponent
        return base

    def atom(self):
        token = self.advance()
        if isinstance(token, int):
            return token
        if token == "(":
            value = self.expr()
            if self.peek() != ")":
                raise ValueError("unbalanced parentheses")
            self.advance()
            return value
        raise ValueError(f"unexpected token {token!r}")


def evaluate(expression):
    parser = _Parser(_tokenize(expression))
    value = parser.expr()
    parser.expect_end()
    return value
