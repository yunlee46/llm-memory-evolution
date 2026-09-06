# Task: RFC 4180 CSV parser

Create a single file `csv_parse.py` exposing `parse(text: str) -> list[list[str]]`.

It parses CSV text into a list of records, each a list of field strings.

## Rules

- Fields are separated by `,`. Records are separated by `\n` or `\r\n`.
- A field may be wrapped in double quotes. Inside a quoted field:
  - `,` and newlines are literal content, not separators;
  - `""` is an escaped literal `"`.
- An unquoted field is taken literally; whitespace is **not** stripped.
- A single trailing record separator at the end of the text does not produce an
  extra empty record. `"a,b\n"` is one record.
- Empty input produces `[]`.
- A blank line in the middle of the text is a record with one empty field.

## Errors

Raise `ValueError` when:

- a quoted field is never closed, e.g. `'"abc'`;
- a character other than `,` or a record separator follows a closing quote,
  e.g. `'"abc"x'`.

## Examples

```
parse("a,b\nc,d")           -> [["a", "b"], ["c", "d"]]
parse('"a,b",c')            -> [["a,b", "c"]]
parse('"say ""hi""",x')     -> [['say "hi"', "x"]]
parse("a,,b")               -> [["a", "", "b"]]
parse("")                   -> []
```

## Constraints

- Standard library only. Do **not** use the `csv` module or `re`.
- The file must be importable as `import csv_parse` from the working directory.
