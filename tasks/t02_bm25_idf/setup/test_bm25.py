import math
from bm25 import bm25_idf


def test_common_term():
    # N=100, df=50: ln((100-50+0.5)/(50+0.5)+1) = ln(1+1)=ln(2)
    assert abs(bm25_idf(100, 50) - math.log(2.0)) < 1e-9

def test_rare_term():
    # df=1: ln((99.5)/(1.5)+1) = ln(67.333..)
    assert abs(bm25_idf(100, 1) - math.log(99.5 / 1.5 + 1)) < 1e-9

def test_df_zero_no_crash():
    v = bm25_idf(100, 0)
    assert v > 0

def test_positive():
    for df in range(0, 101, 10):
        assert bm25_idf(100, df) > 0
