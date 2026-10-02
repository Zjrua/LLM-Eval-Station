"""指数退避重试。"""
import time


def retry_with_backoff(fn, max_attempts=3, base=1.0, sleep=None):
    raise NotImplementedError  # TODO
