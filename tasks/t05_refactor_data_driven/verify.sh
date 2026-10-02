#!/usr/bin/env bash
# 判定+回归。防作弊口径: classifier.py 不得含「硬编码关键词字面量」(应活在 intents.json 里)
set -e
python -m pytest test_classifier.py -q >/dev/null 2>&1 || exit 1
# 行为回归: 分类行为不变
python - <<'PY' || exit 1
from classifier import classify_intent
assert classify_intent("帮我加一个节点") == "mutate"
assert classify_intent("什么是注意力机制") == "query"
assert classify_intent("帮我写一份周报") == "produce"
assert classify_intent("你好呀") == "chat"
PY
# 防硬编码: 关键词字面量不得残留在 classifier.py（"帮我写"等只能出现在 intents.json）
! grep -q "帮我写\|什么是\|加一个" classifier.py || exit 1
[ -f intents.json ] || exit 1
exit 0
