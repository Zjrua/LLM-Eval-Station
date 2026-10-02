"""学习统计。"""
from progress import studylog  # bug: import 错误


def summary(log):
    return {"total_days": len(log.days())}
