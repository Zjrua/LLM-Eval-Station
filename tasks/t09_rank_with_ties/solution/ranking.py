"""榜单排名（竞争排名，带并列）。"""


def rank(scores: dict) -> dict:
    order = sorted(scores, key=lambda k: -scores[k])
    out = {}
    prev_score, prev_rank = None, 0
    for i, name in enumerate(order, 1):
        s = scores[name]
        r = prev_rank if s == prev_score else i
        out[name] = r
        prev_score, prev_rank = s, r
    return out
