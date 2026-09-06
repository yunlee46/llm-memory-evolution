"""Reference solution for interval_ops."""


def merge(intervals):
    cleaned = sorted(
        (int(start), int(end)) for start, end in intervals if start < end
    )
    merged = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def intersect(a, b):
    left = merge(a)
    right = merge(b)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def subtract(a, b):
    result = []
    holes = merge(b)
    for start, end in merge(a):
        cursor = start
        for hole_start, hole_end in holes:
            if hole_end <= cursor or hole_start >= end:
                continue
            if hole_start > cursor:
                result.append((cursor, hole_start))
            cursor = max(cursor, hole_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result
