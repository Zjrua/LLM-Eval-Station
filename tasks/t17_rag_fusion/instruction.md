# 任务：实现三路检索分数融合

`fusion.py` 中 `fuse(scores_by_source, weights)` 未实现。输入：

    scores_by_source = {"bm25": {"doc1": 2.0, "doc2": 1.0},
                        "vector": {"doc1": 0.8, "doc3": 0.6}}
    weights = {"bm25": 0.6, "vector": 0.4}

输出每篇文档的融合分（各源分数 × 源权重求和；文档在某源缺失时按 0 计）。再按融合分**降序**返回 `[(doc, score)]` 列表（分数相同按 doc 字典序）。

要求：`python -m pytest test_fusion.py -q` 全过（不可改）
