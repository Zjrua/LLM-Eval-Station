from schedule import merge


def test_basic_overlap():
    assert merge([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]

def test_touching_merge():
    assert merge([(1, 3), (3, 5)]) == [(1, 5)]

def test_unsorted_input():
    assert merge([(5, 8), (1, 3), (2, 4)]) == [(1, 4), (5, 8)]

def test_contained():
    assert merge([(1, 10), (2, 3)]) == [(1, 10)]

def test_disjoint():
    assert merge([(1, 2), (4, 6), (9, 10)]) == [(1, 2), (4, 6), (9, 10)]

def test_empty_and_single():
    assert merge([]) == []
    assert merge([(1, 1)]) == [(1, 1)]
