# 任务：修复滑窗分块器的 overlap 实现

`chunker.py` 的 `chunk_text(text, size, overlap)` 应把长文本切成重叠分块：每块长 `size`，相邻块起点间隔 `size - overlap`。

已知 bug：当前实现的步长写成了 `size + overlap`（相邻块之间有缝隙，中间字符永远不出现在任何块里）。

要求：
1. 修复步长，保证：每个字符至少出现在一个块里；相邻块重叠区长度恰为 overlap；最后不足 size 的尾巴单独成块
2. `python -m pytest test_chunker.py -q` 全过（不可改）
