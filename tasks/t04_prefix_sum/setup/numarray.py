"""可更新区间和（接口见 instruction）。"""


class NumArray:
    def __init__(self, nums):
        raise NotImplementedError  # TODO

    def update(self, index: int, val: int) -> None:
        raise NotImplementedError  # TODO

    def sum_range(self, left: int, right: int) -> int:
        raise NotImplementedError  # TODO
