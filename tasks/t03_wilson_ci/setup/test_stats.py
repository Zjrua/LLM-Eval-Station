from stats import wilson_ci


def test_basic_contains_p():
    lo, hi = wilson_ci(0.5, 100)
    assert lo < 0.5 < hi

def test_known_value():
    # p=0.7, n=10, z=1.96: lo≈0.397, hi≈0.897（教科书值，容差 0.01）
    lo, hi = wilson_ci(0.7, 10)
    assert abs(lo - 0.3967) < 0.01 and abs(hi - 0.8961) < 0.01

def test_zero_n():
    assert wilson_ci(0.5, 0) == (0.0, 0.0)

def test_bounds():
    lo, hi = wilson_ci(0.0, 10)
    assert lo == 0.0 and 0 <= hi <= 1
    lo, hi = wilson_ci(1.0, 10)
    assert 0 <= lo <= 1 and hi == 1.0

def test_smaller_n_wider():
    lo1, hi1 = wilson_ci(0.5, 10)
    lo2, hi2 = wilson_ci(0.5, 1000)
    assert (hi1 - lo1) > (hi2 - lo2)
