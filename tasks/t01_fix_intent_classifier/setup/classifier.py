"""意图分类器：chat / query / mutate / produce 四类。"""
QUERY_WORDS = ("什么是", "为什么", "怎么", "如何", "解释", "查一下")
MUTATE_WORDS = ("加一个", "删除", "修改", "标记为", "移到", "更新")
PRODUCE_WORDS = ("帮我写", "生成", "总结", "写一份", "产出")


def classify_intent(text: str) -> str:
    """按关键词优先级分类意图。

    优先级：mutate > produce > query > chat。
    """
    for w in MUTATE_WORDS:
        if w in text:
            return "mutate"
    for w in QUERY_WORDS:
        if w in text:
            return "query"
    # bug: 产出类关键词判断时误用了 chat 兜底
    for w in PRODUCE_WORDS:
        if w in text:
            return "chat"
    return "chat"
