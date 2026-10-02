# 任务：实现 BM25 的 IDF 计算

工作目录中的 `bm25.py` 定义了 `bm25_idf(N, df)` 函数但未实现。BM25 采用概率 IDF：

idf(term) = ln((N - df + 0.5) / (df + 0.5) + 1)

其中 N 为文档总数，df 为包含该词的文档数。+1 在分母项内（Lucene 风格），保证 idf 恒为正、df=0 不除零。

要求：
1. 实现 `bm25_idf`，返回 float
2. 通过 `python -m pytest test_bm25.py -q`（该文件不可修改，含数值断言）
