# 任务：实现 Wilson 置信区间

工作目录中的 `stats.py` 定义了 `wilson_ci(p, n, z)` 未实现。Wilson score 区间：

center = (p + z²/2n) / (1 + z²/n)
halfwidth = z·sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)
区间 = [max(0, center-halfwidth), min(1, center+halfwidth)]

要求：
1. 实现 `wilson_ci(p, n, z=1.96)` 返回 (lo, hi) 元组
2. 通过 `python -m pytest test_stats.py -q`
3. 边界：p=0 或 p=1 时区间端点不得越界 [0,1]；n=0 时返回 (0.0, 0.0)
