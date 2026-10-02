"""可更新区间和（线段树实现）。"""


class NumArray:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (2 * self.n)
        for i, v in enumerate(nums):
            self.tree[self.n + i] = v
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, index: int, val: int) -> None:
        i = self.n + index
        diff = val - self.tree[i]
        while i >= 1:
            self.tree[i] += diff
            i //= 2

    def sum_range(self, left: int, right: int) -> int:
        res = 0
        l, r = self.n + left, self.n + right
        while l <= r:
            if l % 2 == 1:
                res += self.tree[l]; l += 1
            if r % 2 == 0:
                res += self.tree[r]; r -= 1
            l //= 2; r //= 2
        return res
