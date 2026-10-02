import pytest
from metrics import success_rate


def test_exact():
    assert success_rate(10, 5) == 50.0

def test_fraction():
    assert success_rate(3, 1) == 33.33
    assert success_rate(3, 2) == 66.67

def test_zero_and_full():
    assert success_rate(7, 0) == 0.0
    assert success_rate(7, 7) == 100.0

def test_round2():
    assert success_rate(6, 1) == 16.67

def test_zero_n_raises():
    with pytest.raises(ZeroDivisionError):
        success_rate(0, 0)
