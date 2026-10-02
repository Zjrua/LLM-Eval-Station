"""意图分类器（数据驱动）：关键词与优先级从 intents.json 加载。"""
import json
from pathlib import Path

_CFG = json.loads((Path(__file__).parent / "intents.json").read_text())
_PRIORITY = ["mutate", "produce", "query"]


def classify_intent(text: str) -> str:
    for intent in _PRIORITY:
        for w in _CFG[intent]:
            if w in text:
                return intent
    return "chat"
