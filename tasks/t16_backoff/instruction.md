# 任务：实现指数退避重试器

`backoff.py` 中 `retry_with_backoff(fn, max_attempts=3, base=1.0, sleep=None)` 未实现。语义：

- 依次调用 fn()，成功（不抛异常）立即返回其返回值
- 抛异常则 sleep(base ** 已失败次数) 后重试，最多 max_attempts 次
- 全部失败：抛出最后一次的异常
- sleep 参数注入（默认 time.sleep），测试用它统计退避时长；每次失败后调用一次 sleep(时长)

要求：`python -m pytest test_backoff.py -q` 全过（不可改）
