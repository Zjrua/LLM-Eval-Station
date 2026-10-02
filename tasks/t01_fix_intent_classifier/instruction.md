# 任务：修复意图分类函数

工作目录中的 `classifier.py` 有一个 `classify_intent(text)` 函数，将用户消息分为四类意图：`chat` / `query` / `mutate` / `produce`。

已知 bug：函数中对 `produce` 类意图（用户要求产出文档/笔记/周报）的关键词判断写错了——当消息包含「帮我写」「生成」「总结」等产出类关键词时，函数会错误地返回 `chat`。

要求：
1. 修复该 bug：包含产出类关键词（帮我写/生成/总结/写一份/产出）的消息应返回 `produce`
2. 保持其余三类意图的现有行为不变（运行 `python -m pytest test_classifier.py -q` 应全部通过，该文件不可修改）
