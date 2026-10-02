# 任务：实现 LRU 缓存

`lru.py` 中 `LRUCache` 未实现（LeetCode 146 接口）：

    LRUCache(capacity)
    .get(key) -> int      # 不存在返回 -1；命中则提升为最近使用
    .put(key, value)      # 淘汰最久未使用；容量满时逐出；put 也算使用

要求：get/put 均摊 O(1)（用哈希+双向链表；O(n) 实现过不了性能用例）。
`python -m pytest test_lru.py -q` 全过（不可改）
