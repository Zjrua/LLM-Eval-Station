# 任务：修复配置合并的三个 bug

`config.py` 的 `merge(base, override)` 做递归合并（override 优先）。已知三个 bug：

1. 递归 dict 合并时直接改了 base 的嵌套 dict（输入被意外修改）
2. 非 dict 值（如 list）被直接共享引用而非复制——修改返回值会影响输入
3. override 中为 None 的值应该被忽略（保留 base），当前 None 覆盖了 base

要求：
1. 修复三个 bug：merge 不修改任何输入；返回值与输入无共享可变结构；None 不覆盖
2. `python -m pytest test_config.py -q` 全过（不可改）
