"""BT 模型测试 (需 numpy + numpyro).

这些测试较慢 (NUTS 采样), 用小样本 + 少采样数加速.
标记 requires_numpyro, 无依赖时自动跳过.
"""

from conftest import requires_numpyro
from llm_eval.models import Dimension, Profile
from llm_eval.learning.model import (
    PriorSpec,
    FitResult,
    fit,
    predict_proba,
    predict_with_fixed_weights,
    weights_to_logits,
)


@requires_numpyro
def test_weights_to_logits_roundtrip():
    """权重 → logits 应保留相对大小关系."""
    import numpy as np
    w = [0.3, 0.1, 0.4, 0.2]
    logits = weights_to_logits(w)
    # 大权重 → 大 logit
    assert logits[2] > logits[0] > logits[3] > logits[1]


@requires_numpyro
def test_fit_basic_convergence():
    """基本拟合: 合成数据应收敛, R-hat < 1.05."""
    import numpy as np
    np.random.seed(42)
    # 构造线性可分数据: y = (x1 - x2) > 0
    n = 200
    X = np.random.randn(n, 2).astype(np.float32)
    # 第一个特征有信号
    logits = 3.0 * X[:, 0] + 0.5 * X[:, 1]
    y = (logits > 0).astype(int)
    # 加点噪声
    flip = np.random.rand(n) < 0.1
    y[flip] = 1 - y[flip]

    prior = PriorSpec(mean_logits=[0.0, 0.0], sigma=0.8)
    result = fit(
        X.tolist(), y.tolist(), prior,
        num_warmup=200, num_samples=400, num_chains=2, seed=0,
        feature_names=["x1", "x2"],
    )
    assert result.converged
    assert result.r_hat < 1.05
    # 学出的权重: x1 应显著大于 x2 (信号在 x1)
    mean = result.posterior_mean()
    assert mean[0] > mean[1]


@requires_numpyro
def test_predict_proba_range():
    """预测概率应在 [0, 1]."""
    import numpy as np
    np.random.seed(0)
    n = 100
    X = np.random.randn(n, 2).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    prior = PriorSpec(mean_logits=[0.0, 0.0], sigma=0.5)
    result = fit(X.tolist(), y.tolist(), prior,
                 num_warmup=150, num_samples=300, num_chains=2, seed=0)
    # 在新数据上预测
    X_new = np.random.randn(20, 2).astype(np.float32)
    probs = predict_proba(result, X_new.tolist())
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert len(probs) == 20


@requires_numpyro
def test_predict_with_fixed_weights():
    """固定权重预测应在 [0, 1]."""
    feats = [[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]]
    weights = [2.0, -1.0]
    probs = predict_with_fixed_weights(weights, feats)
    assert probs[0] > 0.5  # 2*1 + (-1)*(-1) = 3 > 0
    assert probs[1] < 0.5  # -3 < 0
    assert abs(probs[2] - 0.5) < 1e-6  # 0 → sigmoid(0)=0.5


@requires_numpyro
def test_fit_result_normalized_weights_simplex():
    """归一化权重应落在单纯形 (Σ=1, 全正)."""
    import numpy as np
    np.random.seed(0)
    n = 100
    X = np.random.randn(n, 3).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    prior = PriorSpec(mean_logits=[0.0, 0.0, 0.0], sigma=0.5)
    result = fit(X.tolist(), y.tolist(), prior,
                 num_warmup=150, num_samples=300, num_chains=2, seed=0)
    norm_w = result.normalized_weights_on_simplex()
    assert len(norm_w) == 3
    assert all(w >= 0 for w in norm_w)
    assert abs(sum(norm_w) - 1.0) < 0.01
