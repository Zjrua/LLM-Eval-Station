"""多路检索分数融合。"""


def fuse(scores_by_source: dict, weights: dict) -> list:
    out = {}
    for source, scores in scores_by_source.items():
        w = weights.get(source, 0.0)
        for doc, s in scores.items():
            out[doc] = out.get(doc, 0.0) + s * w
    return sorted(out.items(), key=lambda t: (-t[1], t[0]))
