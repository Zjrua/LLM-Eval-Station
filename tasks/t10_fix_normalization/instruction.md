# 任务：修复 min-max 归一化越界 bug

`normalize.py` 的 `normalize(values)` 做 min-max 归一化到 [0,1]。已知 bug：全相同值时应返回全 0（避免除零），但当前实现在最小值非 0 时会返回错误结果（负数或超过 1）。

要求：
1. 修复后：任意输入，输出全部落在 [0,1]；min 元素→0，max 元素→1；全相同→全 0
2. `python -m pytest test_normalize.py -q` 全过（不可改）
