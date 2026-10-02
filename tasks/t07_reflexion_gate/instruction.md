# 任务：实现 Reflexion 自校验判定器

`reflexion.py` 中 `needs_rerun(draft, checklist)` 未实现。语义：检查 `draft`（str）是否遗漏了 `checklist`（list[str]）中的要点——若存在任一要点在 draft 中「未体现」，则需要续跑（返回 True）。

要点体现的判定：要点字符串（大小写不敏感）或其去空格形式出现在 draft 中，即视为已体现。

要求：
1. 实现 `needs_rerun`，返回 bool
2. `python -m pytest test_reflexion.py -q` 全过（不可改）
