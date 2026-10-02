# 任务：实现网关调用日志聚合

`aggregate.py` 中 `aggregate(logs)` 未实现。输入是调用日志字典列表，每条形如：

    {"model": "glm-5.3", "tokens": 1200, "ok": True}

要求按 model 聚合，返回 dict：`{model: {"calls": 调用次数, "tokens": tokens总和, "ok_rate": 成功率(0~1的float)}}`。ok_rate 保留 4 位小数（round）。

要求：
1. 实现 `aggregate`，空列表返回 `{}`
2. `python -m pytest test_aggregate.py -q` 全过（不可改）
