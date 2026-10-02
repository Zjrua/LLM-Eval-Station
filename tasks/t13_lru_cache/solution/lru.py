"""LRU 缓存（哈希 + 双向链表，均摊 O(1)）。"""


class _Node:
    __slots__ = ("k", "v", "prev", "next")

    def __init__(self, k=0, v=0):
        self.k, self.v = k, v
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map = {}
        self.head, self.tail = _Node(), _Node()   # 哨兵
        self.head.next, self.tail.prev = self.tail, self.head

    def _unlink(self, n):
        n.prev.next, n.next.prev = n.next, n.prev

    def _push_front(self, n):
        n.next, n.prev = self.head.next, self.head
        self.head.next.prev = n
        self.head.next = n

    def get(self, key: int) -> int:
        n = self.map.get(key)
        if not n:
            return -1
        self._unlink(n); self._push_front(n)
        return n.v

    def put(self, key: int, value: int) -> None:
        n = self.map.get(key)
        if n:
            n.v = value
            self._unlink(n); self._push_front(n)
            return
        if len(self.map) == self.cap:
            last = self.tail.prev
            self._unlink(last)
            del self.map[last.k]
        n = _Node(key, value)
        self.map[key] = n
        self._push_front(n)
