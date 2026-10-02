"""Reflexion 自校验：判断草稿是否遗漏要点、需要续跑。"""


def needs_rerun(draft: str, checklist: list) -> bool:
    low = draft.lower()
    compact = draft.replace(" ", "").lower()
    for item in checklist:
        if item.lower() in low:
            continue
        if item.replace(" ", "").lower() in compact:
            continue
        return True
    return False
