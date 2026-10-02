"""学习打卡日志。"""
from datetime import date


class StudyLog:
    def __init__(self):
        self._days = set()

    def add(self, date_str: str):
        y, m, d = map(int, date_str.split("-"))
        self._days.add(date(y, m, d))

    def days(self):
        return sorted(self._days)
