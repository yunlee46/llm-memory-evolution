from interval_ops import intersect, merge, subtract


def test_merge_collapses_a_fully_contained_interval():
    assert merge([(1, 100), (10, 20)]) == [(1, 100)]


def test_intersect_normalizes_unsorted_overlapping_inputs():
    assert intersect([(6, 8), (1, 5), (2, 3)], [(0, 100)]) == [(1, 5), (6, 8)]


def test_subtract_removes_multiple_regions():
    assert subtract([(0, 20)], [(2, 4), (10, 12), (18, 30)]) == [
        (0, 2),
        (4, 10),
        (12, 18),
    ]


def test_subtract_with_touching_boundaries_leaves_input_intact():
    assert subtract([(3, 6)], [(0, 3), (6, 9)]) == [(3, 6)]


def test_operations_do_not_mutate_their_arguments():
    a = [(5, 7), (1, 3)]
    b = [(2, 6)]
    merge(a)
    intersect(a, b)
    subtract(a, b)
    assert a == [(5, 7), (1, 3)]
    assert b == [(2, 6)]
