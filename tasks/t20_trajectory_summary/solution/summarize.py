"""trajectory 摘要统计。"""


def summarize(traj: list) -> dict:
    steps = 0
    tool_calls = 0
    empty_replies = 0
    passed = None
    for e in traj:
        if "step" in e:
            steps = max(steps, e["step"] + 1)
        if "assistant" in e and e["assistant"].startswith("TOOL:"):
            tool_calls += 1
        if e.get("event") == "empty_reply":
            empty_replies += 1
        if e.get("event") == "verify":
            passed = e["passed"]
    return {"steps": steps, "tool_calls": tool_calls,
            "empty_replies": empty_replies, "passed": passed}
