"""Held-out tests. Never shown to the agent, never used for selection."""

import word_freq


def test_empty_text_yields_no_tokens():
    assert word_freq.tokenize("") == []
    assert word_freq.top_n("", 5) == []


def test_separator_only_text_yields_no_tokens():
    assert word_freq.tokenize("... --- !!!") == []


def test_apostrophe_only_token_is_dropped():
    assert word_freq.tokenize("a ' b") == ["a", "b"]
