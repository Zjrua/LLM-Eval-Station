"""CSV → 分场景榜单。"""
import csv
import io


def leaderboard(csv_text: str) -> dict:
    seen = {}
    for row in csv.DictReader(io.StringIO(csv_text.strip())):
        key = (row["scenario"], row["model"])
        if key in seen:
            continue
        seen[key] = float(row["score"])
    out = {}
    for (scenario, model), score in seen.items():
        out.setdefault(scenario, []).append((model, score))
    for s in out:
        out[s].sort(key=lambda t: (-t[1], t[0]))
    return out
