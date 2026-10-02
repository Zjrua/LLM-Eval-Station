"""递归配置合并（override 优先）。"""


def merge(base: dict, override: dict) -> dict:
    out = base  # bug1: 直接引用 base
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            merge(out[k], v)
        else:
            out[k] = v  # bug2: 共享引用; bug3: None 覆盖
    return out
