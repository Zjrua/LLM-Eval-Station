import random
import time
from numarray import NumArray


def test_basic():
    na = NumArray([1, 3, 5])
    assert na.sum_range(0, 2) == 9
    na.update(1, 2)
    assert na.sum_range(0, 2) == 8
    assert na.sum_range(1, 1) == 2

def test_single():
    na = NumArray([-1])
    assert na.sum_range(0, 0) == -1

def test_random_against_bruteforce():
    random.seed(42)
    nums = [random.randint(-100, 100) for _ in range(200)]
    na = NumArray(nums)
    ref = nums[:]
    for _ in range(500):
        if random.random() < 0.5:
            i, v = random.randrange(200), random.randint(-100, 100)
            na.update(i, v); ref[i] = v
        else:
            l, r = sorted(random.sample(range(200), 2))
            assert na.sum_range(l, r) == sum(ref[l:r+1])

def test_performance():
    random.seed(7)
    n = 20000
    nums = [random.randint(-1000, 1000) for _ in range(n)]
    na = NumArray(nums)
    t0 = time.time()
    for _ in range(10000):
        na.update(random.randrange(n), random.randint(-1000, 1000))
        na.sum_range(0, n - 1)
    assert time.time() - t0 < 5.0  # O(n) 暴力约需 30s+，会超时
