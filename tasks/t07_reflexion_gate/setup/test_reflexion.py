from reflexion import needs_rerun


def test_all_covered():
    assert needs_rerun("涵盖甲和乙", ["甲", "乙"]) is False

def test_missing_one():
    assert needs_rerun("只有甲", ["甲", "乙"]) is True

def test_case_insensitive():
    assert needs_rerun("includes API docs", ["api"]) is False

def test_whitespace_insensitive():
    assert needs_rerun("done: 分层检索", ["分层 检索"]) is False

def test_empty_checklist():
    assert needs_rerun("任意", []) is False
