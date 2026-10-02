# 任务：实现区间和查询类 NumArray

工作目录中的 `numarray.py` 定义了 `NumArray` 类未实现（LeetCode 307 区域和检索风格的接口）：

    NumArray(nums)            # 构造，nums 为整数列表
    .update(index, val)       # 把 nums[index] 设为 val
    .sum_range(left, right)   # 返回闭区间 [left, right] 的元素和（含两端）

要求：
1. 实现该类，update 和 sum_range 均需好于 O(n)（建议线段树或树状数组；O(n) 暴力会超时）
2. 通过 `python -m pytest test_numarray.py -q`，其中含一个 50000 次混合作询的性能用例
