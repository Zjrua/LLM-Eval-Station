from lint import lint_markdown


def test_clean_doc():
    doc = "# 标题\n\n## 小结\n- a\n\n## 练习\n- b\n"
    assert lint_markdown(doc) == []

def test_h1_count():
    doc = "# a\n# b\n## 小结\n## 练习\n"
    assert "h1-count:2" in lint_markdown(doc)

def test_interview_hype():
    doc = "# t\n正文\n这是面试要点\n## 小结\n## 练习\n"
    assert "interview-hype:3" in lint_markdown(doc)

def test_section_missing():
    doc = "# t\n## 小结\n"
    problems = lint_markdown(doc)
    assert "section-missing:练习" in problems

def test_section_order():
    doc = "# t\n## 练习\n## 小结\n"
    assert "section-order" in lint_markdown(doc)
