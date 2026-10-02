# 任务：从 CSV 生成分场景榜单

`report.py` 中 `leaderboard(csv_text)` 未实现。输入 CSV 文本（首行表头）：

    model,scenario,score
    glm,coding,0.9
    glm,math,0.7

返回 `{scenario: [(model, score), ...]}`：每个场景内按分数**降序**排列（分数相同按模型名字典序），只保留每个模型在该场景的第一条记录（重复行忽略）。

要求：`python -m pytest test_report.py -q` 全过（不可改）
