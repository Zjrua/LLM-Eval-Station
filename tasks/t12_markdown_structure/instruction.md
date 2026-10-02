# 任务：实现教材 notebook 导出的 Markdown 结构校验

`lint.py` 中 `lint_markdown(text)` 未实现。校验规则（返回问题列表，每个问题一行字符串；无问题返回 []）：

1. 一级标题（`# `开头行）恰好 1 个，否则报 "h1-count:<实际数量>"
2. 不允许出现「面试」字样，每出现一行报一个 "interview-hype:<行号>"（1-based）
3. 「小结」节（## 小结）必须在「练习」节（## 练习）之前，缺失任一报 "section-missing:小结" 或 "section-missing:练习"，顺序反了报 "section-order"

要求：`python -m pytest test_lint.py -q` 全过（不可改）
