"""Markdown 结构校验。"""


def lint_markdown(text: str) -> list:
    lines = text.split("\n")
    problems = []
    h1 = sum(1 for l in lines if l.startswith("# "))
    if h1 != 1:
        problems.append(f"h1-count:{h1}")
    for i, l in enumerate(lines, 1):
        if "面试" in l:
            problems.append(f"interview-hype:{i}")
    has_summary = any(l.strip() == "## 小结" for l in lines)
    has_exercise = any(l.strip() == "## 练习" for l in lines)
    if not has_summary:
        problems.append("section-missing:小结")
    if not has_exercise:
        problems.append("section-missing:练习")
    if has_summary and has_exercise:
        si = next(i for i, l in enumerate(lines) if l.strip() == "## 小结")
        ei = next(i for i, l in enumerate(lines) if l.strip() == "## 练习")
        if si > ei:
            problems.append("section-order")
    return problems
