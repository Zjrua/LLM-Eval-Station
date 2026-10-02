from ranking import rank


def test_basic():
    assert rank({"a": 90, "b": 80, "c": 90, "d": 70}) == {"a": 1, "c": 1, "b": 3, "d": 4}

def test_all_tied():
    assert rank({"x": 5, "y": 5, "z": 5}) == {"x": 1, "y": 1, "z": 1}

def test_single():
    assert rank({"only": 1.5}) == {"only": 1}

def test_float_scores():
    r = rank({"m1": 0.91, "m2": 0.93, "m3": 0.91})
    assert r["m2"] == 1 and r["m1"] == 2 and r["m3"] == 2
