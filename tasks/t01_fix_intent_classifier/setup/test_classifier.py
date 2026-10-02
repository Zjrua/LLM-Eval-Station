from classifier import classify_intent


def test_mutate_priority():
    assert classify_intent("帮我加一个节点") == "mutate"

def test_query():
    assert classify_intent("什么是注意力机制") == "query"

def test_produce_write():
    assert classify_intent("帮我写一份周报") == "produce"

def test_produce_generate():
    assert classify_intent("生成学习笔记") == "produce"

def test_produce_summary():
    assert classify_intent("总结一下这章内容") == "produce"

def test_chat_fallback():
    assert classify_intent("你好呀") == "chat"
