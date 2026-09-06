"""Reference solution. Must score 1.0 on both visible and held-out suites."""

import re
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z0-9']+")


def tokenize(text):
    tokens = []
    for match in _TOKEN.findall(text or ""):
        token = match.strip("'").lower()
        if token:
            tokens.append(token)
    return tokens


def top_n(text, n):
    if n <= 0:
        return []
    counts = Counter(tokenize(text))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
