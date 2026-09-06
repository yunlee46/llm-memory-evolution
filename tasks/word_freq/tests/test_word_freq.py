"""Visible tests for the word_freq task. Agents never see this file."""

import word_freq


def test_tokenize_lowercases():
    assert word_freq.tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_splits_on_punctuation():
    assert word_freq.tokenize("a,b;c!d") == ["a", "b", "c", "d"]


def test_tokenize_keeps_internal_apostrophes():
    assert word_freq.tokenize("don't stop") == ["don't", "stop"]


def test_tokenize_strips_edge_apostrophes():
    assert word_freq.tokenize("'quoted' word") == ["quoted", "word"]


def test_tokenize_keeps_digits():
    assert word_freq.tokenize("route 66 now") == ["route", "66", "now"]


def test_top_n_orders_by_count_then_alphabetically():
    text = "b a a c b a"
    assert word_freq.top_n(text, 3) == [("a", 3), ("b", 2), ("c", 1)]


def test_top_n_larger_than_vocabulary_returns_all():
    assert word_freq.top_n("x y", 99) == [("x", 1), ("y", 1)]


def test_top_n_non_positive_returns_empty():
    assert word_freq.top_n("a b c", 0) == []
    assert word_freq.top_n("a b c", -1) == []
