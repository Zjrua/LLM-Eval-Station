"""BM25 评分的 IDF 组件（Lucene 风格概率 IDF）。"""
import math


def bm25_idf(N: int, df: int) -> float:
    """N: 文档总数；df: 包含该词的文档数。

    idf = ln((N - df + 0.5) / (df + 0.5) + 1)
    """
    return math.log((N - df + 0.5) / (df + 0.5) + 1)
