"""min-max 归一化到 [0,1]。"""


def normalize(values):
    lo, hi = min(values), max(values)
    # bug: 未处理全相同（hi==lo 时除零）；且未夹紧
    return [(v - lo) / (hi - lo) for v in values]
