from aggregate import aggregate


def test_basic():
    logs = [
        {"model": "a", "tokens": 100, "ok": True},
        {"model": "a", "tokens": 50, "ok": False},
        {"model": "b", "tokens": 10, "ok": True},
    ]
    r = aggregate(logs)
    assert r["a"]["calls"] == 2 and r["a"]["tokens"] == 150 and r["a"]["ok_rate"] == 0.5
    assert r["b"]["calls"] == 1 and r["b"]["ok_rate"] == 1.0

def test_round_4():
    logs = [{"model": "m", "tokens": 1, "ok": True}] * 3 + [{"model": "m", "tokens": 1, "ok": False}]
    assert aggregate(logs)["m"]["ok_rate"] == 0.75

def test_empty():
    assert aggregate([]) == {}
