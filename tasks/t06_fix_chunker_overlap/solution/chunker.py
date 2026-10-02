"""滑窗文本分块（RAG chunking）。"""


def chunk_text(text: str, size: int = 10, overlap: int = 3):
    """返回分块列表。相邻块起点间隔应为 size - overlap。"""
    if size <= overlap:
        raise ValueError("size must be > overlap")
    chunks = []
    step = size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + size])
        if i + size >= len(text):
            break
    return chunks
