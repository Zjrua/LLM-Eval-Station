import copy
from config import merge


BASE = {"a": 1, "nested": {"x": 1, "y": 2}, "list": [1, 2]}


def test_basic_override():
    assert merge({"a": 1}, {"a": 2})["a"] == 2

def test_nested_merge():
    r = merge({"n": {"x": 1, "y": 2}}, {"n": {"y": 3}})
    assert r["n"] == {"x": 1, "y": 3}

def test_inputs_not_mutated():
    base = copy.deepcopy(BASE)
    merge(base, {"nested": {"z": 9}, "new": 5})
    assert base == BASE

def test_no_shared_structure():
    base = copy.deepcopy(BASE)
    r = merge(base, {"b": 2})
    r["list"].append(99)
    r["nested"]["x"] = 100
    assert base == BASE

def test_none_keeps_base():
    r = merge({"a": 1}, {"a": None})
    assert r["a"] == 1
