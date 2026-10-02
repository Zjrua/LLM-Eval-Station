"""比例的 Wilson score 置信区间。"""
import math


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple:
    """返回 (lo, hi)，n=0 时返回 (0.0, 0.0)。"""
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
