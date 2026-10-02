import random
import time
from lru import LRUCache


def test_lc_example():
    c = LRUCache(2)
    c.put(1, 1); c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)                      # 逐出 key 2
    assert c.get(2) == -1
    c.put(4, 4)                      # 逐出 key 1
    assert c.get(1) == -1
    assert c.get(3) == 3 and c.get(4) == 4

def test_update_refresh():
    c = LRUCache(2)
    c.put(1, 1); c.put(2, 2)
    c.put(1, 10)                     # 更新 1，1 变最近
    c.put(3, 3)                      # 逐出 2
    assert c.get(2) == -1 and c.get(1) == 10

def test_capacity_one():
    c = LRUCache(1)
    c.put(1, 1); c.put(2, 2)
    assert c.get(1) == -1 and c.get(2) == 2

def test_performance():
    random.seed(1)
    c = LRUCache(1000)
    t0 = time.time()
    for i in range(100000):
        k = random.randrange(1200)
        if random.random() < 0.5:
            c.put(k, i)
        else:
            c.get(k)
    assert time.time() - t0 < 4.0    # O(n) 实现约需 20s+
