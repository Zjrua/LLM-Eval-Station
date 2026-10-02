"""按模型聚合调用日志。"""


def aggregate(logs: list) -> dict:
    out = {}
    for l in logs:
        m = out.setdefault(l["model"], {"calls": 0, "tokens": 0, "ok": 0})
        m["calls"] += 1
        m["tokens"] += l["tokens"]
        m["ok"] += 1 if l["ok"] else 0
    for m in out.values():
        m["ok_rate"] = round(m.pop("ok") / m["calls"], 4)
    return out
