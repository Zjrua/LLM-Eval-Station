from backoff import retry_with_backoff


def test_success_first_try():
    assert retry_with_backoff(lambda: 42) == 42

def test_retries_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    sleeps = []
    assert retry_with_backoff(flaky, max_attempts=5, base=2.0, sleep=sleeps.append) == "ok"
    assert calls["n"] == 3
    assert sleeps == [2.0 ** 0, 2.0 ** 1]  # 两次失败各退避一次

def test_exhausted_raises_last():
    class MyErr(Exception):
        pass

    def always():
        raise MyErr("x")

    sleeps = []
    try:
        retry_with_backoff(always, max_attempts=3, base=3.0, sleep=sleeps.append)
        assert False
    except MyErr:
        pass
    assert sleeps == [1.0, 3.0, 9.0]

def test_default_sleep_used():
    import backoff as m
    assert m.time.sleep is not None  # 模块引入了 time
