"""意图分类器：chat / query / mutate / produce 四类。"""
QUERY_WORDS = ("什么是", "为什么", "怎么", "如何", "解释", "查一下")
MUTATE_WORDS = ("加一个", "删除", "修改", "标记为", "移到", "更新")
PRODUCE_WORDS = ("帮我写", "生成", "总结", "写一份", "产出")


def classify_intent(text: str) -> str:
    for w in MUTATE_WORDS:
        if w in text:
            return "mutate"
    for w in QUERY_WORDS:
        if w in text:
            return "query"
    for w in PRODUCE_WORDS:
        if w in text:
            return "produce"
    return "chat"
