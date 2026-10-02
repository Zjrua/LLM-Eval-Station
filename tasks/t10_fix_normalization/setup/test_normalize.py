import math
from normalize import normalize


def test_basic():
    assert normalize([1, 2, 3]) == [0.0, 0.5, 1.0]

def test_negative_shift():
    r = normalize([-10, 0, 10])
    assert r[0] == 0.0 and r[2] == 1.0 and abs(r[1] - 0.5) < 1e-12

def test_all_same():
    assert normalize([7, 7, 7]) == [0.0, 0.0, 0.0]

def test_in_range():
    r = normalize([3, 5, 5, 9, 11])
    assert all(0.0 <= x <= 1.0 for x in r)

def test_single():
    assert normalize([42]) == [0.0]
