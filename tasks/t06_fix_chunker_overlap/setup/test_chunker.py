from chunker import chunk_text


def test_basic_shape():
    chunks = chunk_text("abcdefghij" * 3, size=10, overlap=3)
    assert all(len(c) == 10 for c in chunks[:-1])

def test_full_coverage():
    text = "".join(chr(ord('a') + i % 26) for i in range(100))
    chunks = chunk_text(text, size=10, overlap=3)
    covered = "".join(chunks)
    assert all(ch in covered for ch in set(text))

def test_overlap_exact():
    chunks = chunk_text("0123456789" * 5, size=10, overlap=3)
    for a, b in zip(chunks, chunks[1:]):
        assert a[-3:] == b[:3]  # 相邻块重叠 3 字符

def test_short_text():
    assert chunk_text("abc", size=10, overlap=3) == ["abc"]

def test_tail_chunk():
    chunks = chunk_text("x" * 25, size=10, overlap=3)
    # 滑窗 0,7,14,21：尾块 [21:25] 长 4，全覆盖
    assert len(chunks[-1]) == 4
