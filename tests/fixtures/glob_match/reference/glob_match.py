"""Reference solution for glob_match."""


def _class_end(pattern, start):
    index = start + 1
    if index < len(pattern) and pattern[index] == "!":
        index += 1
    if index < len(pattern) and pattern[index] == "]":
        index += 1
    while index < len(pattern) and pattern[index] != "]":
        index += 1
    if index >= len(pattern):
        raise ValueError(f"unterminated character class at {start}")
    return index


def _class_matches(body, char):
    if char == "/":
        return False
    negated = body.startswith("!")
    if negated:
        body = body[1:]
    matched = False
    index = 0
    while index < len(body):
        if index + 2 < len(body) and body[index + 1] == "-":
            if body[index] <= char <= body[index + 2]:
                matched = True
            index += 3
        else:
            if body[index] == char:
                matched = True
            index += 1
    return matched != negated


def _match(pattern, p, text, t, memo):
    key = (p, t)
    if key in memo:
        return memo[key]

    if p == len(pattern):
        result = t == len(text)
    elif pattern.startswith("**", p):
        result = any(
            _match(pattern, p + 2, text, k, memo) for k in range(t, len(text) + 1)
        )
    elif pattern[p] == "*":
        result = False
        k = t
        while True:
            if _match(pattern, p + 1, text, k, memo):
                result = True
                break
            if k >= len(text) or text[k] == "/":
                break
            k += 1
    elif pattern[p] == "?":
        result = (
            t < len(text)
            and text[t] != "/"
            and _match(pattern, p + 1, text, t + 1, memo)
        )
    elif pattern[p] == "[":
        end = _class_end(pattern, p)
        result = (
            t < len(text)
            and _class_matches(pattern[p + 1 : end], text[t])
            and _match(pattern, end + 1, text, t + 1, memo)
        )
    else:
        result = (
            t < len(text)
            and text[t] == pattern[p]
            and _match(pattern, p + 1, text, t + 1, memo)
        )

    memo[key] = result
    return result


def match(pattern, text):
    # Validate every class up front so an unterminated one raises even when an
    # earlier mismatch would otherwise short-circuit the walk.
    index = 0
    while index < len(pattern):
        if pattern[index] == "[":
            index = _class_end(pattern, index) + 1
        else:
            index += 1
    return _match(pattern, 0, text, 0, {})
