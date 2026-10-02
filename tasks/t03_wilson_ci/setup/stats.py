"""比例的 Wilson score 置信区间。"""
import math


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple:
    """返回 (lo, hi)，n=0 时返回 (0.0, 0.0)。"""
    raise NotImplementedError  # TODO: 实现
