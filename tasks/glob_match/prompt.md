# Task: glob pattern matcher

Create a single file `glob_match.py` exposing `match(pattern: str, text: str) -> bool`.

It reports whether `text` matches `pattern`. The whole text must match, not a prefix.

## Pattern syntax

| Token | Meaning |
|---|---|
| `?` | exactly one character, but never `/` |
| `*` | zero or more characters, but never `/` |
| `**` | zero or more characters, **including** `/` |
| `[abc]` | one character from the set, but never `/` |
| `[a-z]` | one character from the range, but never `/` |
| `[!abc]` | one character *not* in the set, and never `/` |

Any other character matches itself literally.

Inside a character class, a `]` appearing immediately after `[` or `[!` is a literal
`]` rather than the terminator. A `-` at the start or end of a class is a literal `-`.

## Examples

```
match("*.py", "main.py")            -> True
match("*.py", "src/main.py")        -> False    # * does not cross /
match("**/*.py", "src/a/main.py")   -> True
match("**main.py", "main.py")       -> True    # ** may match nothing
match("**/main.py", "main.py")      -> False   # ** is characters, not segments
match("file?.txt", "file1.txt")     -> True
match("[a-c]at", "bat")             -> True
match("[!a-c]at", "bat")            -> False
```

## Errors

Raise `ValueError` if a character class is never closed, e.g. `"[abc"`.

## Constraints

- Standard library only. Do **not** use `fnmatch`, `glob`, `pathlib`, or `re`.
- The file must be importable as `import glob_match` from the working directory.
