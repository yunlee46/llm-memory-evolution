"""Partial solution: never strips edge apostrophes.

Fails exactly 1 of 8 visible tests and 1 of 3 held-out tests.
"""

import re
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z0-9']+")


def tokenize(text):
    return [match.lower() for match in _TOKEN.findall(text or "")]


def top_n(text, n):
    if n <= 0:
        return []
    counts = Counter(tokenize(text))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
