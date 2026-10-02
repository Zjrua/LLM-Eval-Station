from report import leaderboard


CSV = """model,scenario,score
glm,coding,0.9
glm,math,0.7
dsk,coding,0.95
dsk,math,0.6
qwn,coding,0.9
glm,coding,0.1
"""


def test_scenarios():
    r = leaderboard(CSV)
    assert set(r.keys()) == {"coding", "math"}

def test_desc_and_tie_alpha():
    r = leaderboard(CSV)
    assert r["coding"] == [("dsk", 0.95), ("glm", 0.9), ("qwn", 0.9)]

def test_first_row_wins():
    # glm,coding 出现两次（0.9 与 0.1），保留第一条 0.9
    assert ("glm", 0.9) in leaderboard(CSV)["coding"]

def test_math():
    assert leaderboard(CSV)["math"] == [("glm", 0.7), ("dsk", 0.6)]

def test_score_as_float():
    assert all(isinstance(s, float) for _, s in leaderboard(CSV)["coding"])
