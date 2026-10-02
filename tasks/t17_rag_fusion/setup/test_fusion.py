from fusion import fuse


S = {"bm25": {"a": 2.0, "b": 1.0}, "vector": {"a": 0.5, "c": 1.0}}
W = {"bm25": 0.6, "vector": 0.4}


def test_weighted_sum():
    r = dict(fuse(S, W))
    assert abs(r["a"] - (2.0 * 0.6 + 0.5 * 0.4)) < 1e-9

def test_missing_source_zero():
    r = dict(fuse(S, W))
    assert abs(r["b"] - 0.6) < 1e-9      # b 只在 bm25
    assert abs(r["c"] - 0.4) < 1e-9      # c 只在 vector

def test_desc_order_tie_alpha():
    r = fuse({"s": {"x": 1.0, "a": 1.0}}, {"s": 1.0})
    assert [d for d, _ in r] == ["a", "x"]

def test_single_source():
    assert fuse({"s": {"d": 3.0}}, {"s": 0.5}) == [("d", 1.5)]
