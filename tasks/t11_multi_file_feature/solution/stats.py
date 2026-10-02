"""学习统计。"""
from progress import StudyLog  # noqa: F401（保留可用符号）


def summary(log):
    days = log.days()
    streak = 0
    if days:
        streak = 1
        best = 1
        for prev, cur in zip(days, days[1:]):
            streak = streak + 1 if (cur - prev).days == 1 else 1
            best = max(best, streak)
        streak = best
    return {"total_days": len(days), "streak": streak}
