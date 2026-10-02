from summarize import summarize


def test_basic():
    t = [
        {"step": 0, "assistant": "TOOL: read({})"},
        {"step": 0, "observation": "..."},
        {"step": 1, "assistant": "TOOL: bash({})"},
        {"step": 1, "observation": "..."},
        {"event": "verify", "passed": True},
    ]
    s = summarize(t)
    assert s == {"steps": 2, "tool_calls": 2, "empty_replies": 0, "passed": True}

def test_empty_reply():
    t = [
        {"step": 0, "assistant": "thinking..."},
        {"step": 0, "event": "empty_reply"},
        {"step": 1, "assistant": "TOOL: write({})"},
        {"event": "verify", "passed": False},
    ]
    s = summarize(t)
    assert s["empty_replies"] == 1 and s["tool_calls"] == 1 and s["passed"] is False

def test_no_verify():
    assert summarize([{"step": 0, "assistant": "hi"}])["passed"] is None

def test_empty():
    assert summarize([]) == {"steps": 0, "tool_calls": 0, "empty_replies": 0, "passed": None}
