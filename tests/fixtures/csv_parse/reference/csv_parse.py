"""Reference solution for csv_parse."""


def parse(text):
    if not text:
        return []

    records = []
    fields = []
    field = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char == '"':
            index += 1
            closed = False
            while index < length:
                if text[index] == '"':
                    if index + 1 < length and text[index + 1] == '"':
                        field.append('"')
                        index += 2
                        continue
                    index += 1
                    closed = True
                    break
                field.append(text[index])
                index += 1
            if not closed:
                raise ValueError("unterminated quoted field")
            if index < length and text[index] not in (",", "\n", "\r"):
                raise ValueError(f"unexpected character after closing quote at {index}")
            continue

        if char == ",":
            fields.append("".join(field))
            field = []
            index += 1
            continue

        if char == "\r" and index + 1 < length and text[index + 1] == "\n":
            fields.append("".join(field))
            records.append(fields)
            fields, field = [], []
            index += 2
            continue

        if char == "\n":
            fields.append("".join(field))
            records.append(fields)
            fields, field = [], []
            index += 1
            continue

        field.append(char)
        index += 1

    # A single trailing record separator closes the last record rather than
    # opening an empty one. Anything still buffered means it did not.
    if field or fields or not text.endswith(("\n", "\r")):
        fields.append("".join(field))
        records.append(fields)
    return records
