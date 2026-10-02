import json
import classifier


def test_behavior_unchanged():
    assert classifier.classify_intent("帮我加一个节点") == "mutate"
    assert classifier.classify_intent("什么是注意力机制") == "query"
    assert classifier.classify_intent("帮我写一份周报") == "produce"
    assert classifier.classify_intent("你好呀") == "chat"

def test_data_file_exists_and_valid():
    d = json.load(open("intents.json"))
    assert set(d.keys()) == {"chat", "query", "mutate", "produce"}
    for k, v in d.items():
        assert isinstance(v, list) and len(v) >= 3

def test_new_keyword_no_code_change():
    d = json.load(open("intents.json"))
    if "汇报" not in d["produce"]:
        d["produce"].append("汇报")
        json.dump(d, open("intents.json", "w"), ensure_ascii=False)
    import importlib
    importlib.reload(classifier)
    assert classifier.classify_intent("汇报一下进度") == "produce"
