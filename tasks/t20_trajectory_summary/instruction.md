# 任务：实现 agent trajectory 摘要统计

`summarize.py` 中 `summarize(traj)` 未实现。输入是事件列表（本 harness 的 trajectory JSON 格式的子集）：

    [{"step": 0, "assistant": "TOOL: read(...)"},
     {"step": 0, "observation": "file content..."},
     {"step": 1, "assistant": ""},
     {"step": 1, "event": "empty_reply"},
     {"event": "verify", "passed": True}]

返回 dict：`{"steps": 轮数(出现过的最大 step+1，无 step 键的事件不算), "tool_calls": assistant 消息中以 "TOOL:" 开头的数量, "empty_replies": event 为 empty_reply 的数量, "passed": 最后一个 verify 事件的 passed 值(无 verify 则 None)}`。

要求：`python -m pytest test_summarize.py -q` 全过（不可改）
