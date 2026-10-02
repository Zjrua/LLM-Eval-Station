# 任务：把硬编码意图分类器重构为数据驱动

`classifier.py` 中意图关键词硬编码在函数外的四个元组里。要求重构：

1. 新建 `intents.json`，格式：`{"chat": [...], "query": [...], "mutate": [...], "produce": [...]}`，把现有关键词迁移进去（优先级顺序也要保留：mutate > produce > query > chat）
2. 改写 `classifier.py`：从 `intents.json` 加载关键词（模块级加载一次），`classify_intent` 行为与现在完全一致
3. `python -m pytest test_classifier.py -q` 必须全过（该文件不可改）
4. 新增行为：`intents.json` 里加一个新关键词（如 "汇报"→produce），不改 classifier.py 也能生效（测试有检验此项）
