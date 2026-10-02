"""指数退避重试。"""
import time


def retry_with_backoff(fn, max_attempts=3, base=1.0, sleep=None):
    if sleep is None:
        sleep = time.sleep
    last = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep(base ** attempt)
    raise last
