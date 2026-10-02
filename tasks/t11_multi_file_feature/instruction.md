# 任务：给学习进度模块加"连续学习天数"统计

工作目录是一个小包：
- `progress.py`：`StudyLog` 类，`add(date_str)` 讇记某日学习（ISO 日期字符串，可重复 add 同一天）
- `stats.py`：`summary(log)` 返回 {"total_days": n}

要求：
1. 修正 `stats.py` 中错误的 import（当前 `from progress import studylog` 找不到符号，运行即崩）
2. 在 `stats.py` 的 `summary` 里增加 `"streak"`：最长连续学习天数（按日期自然日连续，与 add 顺序无关）
3. `python -m pytest test_stats.py -q` 全过（不可改）
