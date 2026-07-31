"""pytest 共享配置.

- 注入 src/ 到 sys.path, 使测试无需安装包即可 import llm_eval.*
- 提供共享 fixtures (维度分模型池、合成 pairwise 样本)
"""

import sys
from pathlib import Path

# 让 tests/ 与 scripts/*.py 一致: 从源码树直接导入
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 标记需要 learning extra 的测试; 未安装依赖时自动跳过
import pytest


def _has_learning_deps() -> bool:
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


requires_learning = pytest.mark.skipif(
    not _has_learning_deps(),
    reason="需要 `pip install -e '.[learning]'` 安装 numpy/scikit-learn/jax/numpyro",
)


# 同样需要 NumPyro/JAX 的更重测试
def _has_numpyro() -> bool:
    try:
        import numpyro  # noqa: F401
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


requires_numpyro = pytest.mark.skipif(
    not _has_numpyro(),
    reason="需要 numpyro/jax (属于 [learning] extra)",
)
