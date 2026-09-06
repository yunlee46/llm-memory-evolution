# Task: word frequency module

Create a single file `word_freq.py` exposing two functions.

## `tokenize(text: str) -> list[str]`

Split `text` into lowercase word tokens.

- A word is a run of letters, digits, and apostrophes.
- Everything else is a separator.
- Leading and trailing apostrophes are stripped from each token.
- Tokens that end up empty are dropped.

## `top_n(text: str, n: int) -> list[tuple[str, int]]`

Return the `n` most frequent tokens as `(word, count)` pairs.

- Sorted by count descending, then by word ascending for ties.
- If `n` is greater than the number of distinct words, return all of them.
- If `n <= 0`, return an empty list.
- An empty or separator-only `text` yields an empty list.

## Constraints

- Standard library only.
- The file must be importable as `import word_freq` from the working directory.
