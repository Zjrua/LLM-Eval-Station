"""成功率的百分比表示。"""


def success_rate(n: int, k: int) -> float:
    """k 成功 / n 总数, 百分比 0~100, 2 位小数。"""
    if n == 0:
        raise ZeroDivisionError("n must be > 0")
    return round(k // n * 100, 2)  # bug: 整除截断
