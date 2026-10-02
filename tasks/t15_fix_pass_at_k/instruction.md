# 任务：修复正确率计算被整除截断的 bug

`metrics.py` 的 `success_rate(n, k)` 计算 k/n 的百分比（0~100 的 float，保留 2 位小数）。已知 bug：实现用了整除 `//`，当 k 不整除 n 时结果被截断成整数百分比（如 1/3 应为 33.33，得到 0.0）。

要求：
1. 修复为真除法并 round 到 2 位小数
2. n=0 时抛 `ZeroDivisionError`
3. `python -m pytest test_metrics.py -q` 全过（不可改）
