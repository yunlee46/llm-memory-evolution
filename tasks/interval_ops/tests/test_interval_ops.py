from interval_ops import intersect, merge, subtract


def test_merge_sorts_and_combines_overlaps():
    assert merge([(5, 7), (1, 3), (2, 4)]) == [(1, 4), (5, 7)]


def test_merge_combines_touching_intervals():
    assert merge([(1, 3), (3, 5)]) == [(1, 5)]


def test_merge_drops_empty_intervals():
    assert merge([(4, 4), (1, 2), (7, 3)]) == [(1, 2)]


def test_merge_of_empty_list_is_empty():
    assert merge([]) == []


def test_merge_returns_tuples():
    assert all(isinstance(item, tuple) for item in merge([(1, 2)]))


def test_intersect_picks_shared_regions():
    assert intersect([(1, 10)], [(2, 4), (6, 8)]) == [(2, 4), (6, 8)]


def test_intersect_of_disjoint_sets_is_empty():
    assert intersect([(1, 3)], [(5, 7)]) == []


def test_intersect_at_a_touching_boundary_is_empty():
    assert intersect([(1, 3)], [(3, 5)]) == []


def test_subtract_punches_a_hole():
    assert subtract([(1, 10)], [(3, 5)]) == [(1, 3), (5, 10)]


def test_subtract_everything_is_empty():
    assert subtract([(1, 5)], [(0, 10)]) == []


def test_subtract_nothing_returns_normalized_input():
    assert subtract([(5, 7), (1, 3)], []) == [(1, 3), (5, 7)]
