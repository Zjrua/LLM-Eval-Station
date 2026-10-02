from progress import StudyLog
from stats import summary


def test_total():
    log = StudyLog()
    log.add("2026-09-01"); log.add("2026-09-02"); log.add("2026-09-01")
    assert summary(log)["total_days"] == 2

def test_streak():
    log = StudyLog()
    for d in ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-07", "2026-09-08"]:
        log.add(d)
    s = summary(log)
    assert s["streak"] == 3

def test_streak_order_independent():
    log = StudyLog()
    for d in ["2026-09-08", "2026-09-01", "2026-09-03", "2026-09-02"]:
        log.add(d)
    assert summary(log)["streak"] == 3

def test_empty():
    log = StudyLog()
    s = summary(log)
    assert s["total_days"] == 0 and s["streak"] == 0
