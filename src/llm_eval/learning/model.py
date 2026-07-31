"""Bradley-Terry 贝叶斯模型 (NumPyro 实现).

模型: P(A beats B) = σ(w · (x_A - x_B))
    其中 x 是潜因子特征 (可能含交叉项), w 是权重向量.

设计 (对应"可解释 + 尊重小数据"约束):
    - 先验: Logistic-Normal. 把手工权重转成 logits 作先验均值, σ 控制锚定强度
    - σ 小 (0.3): 强先验, 数据少时退化为手工权重 (安全)
    - σ 大 (0.8): 弱先验, 数据多时数据主导
    - 推断: NUTS 采样, 得真后验 → 天然带 CI
    - 收敛诊断: R-hat < 1.01, ESS > 400

为什么不直接用 softmax(w) 落单纯形:
    BT 模型对 w 的**绝对尺度不敏感** (只关心 w·diff 的符号和相对大小),
    所以不需要 w 落单纯形. 但报告时我们会把 w 归一化到单纯形展示 (便于和手工权重对比).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 在 JAX 初始化前配置足够的虚拟设备, 让多链 MCMC 可并行 (而非顺序跑 + warning).
# 必须在首次 import jax 后立即设; 4 个 host device 足够默认的 2 链.
try:
    import numpyro
    numpyro.set_host_device_count(4)
except Exception:  # noqa: BLE001
    pass  # numpyro 未装 (核心包用法) 时静默跳过


@dataclass
class PriorSpec:
    """先验规格."""

    mean_logits: list[float]  # 先验均值 (logits 空间)
    sigma: float = 0.5  # 先验标准差; 0.3=强锚定, 0.5=中等, 0.8=弱
    use_horseshoe: bool = False  # 是否加 horseshoe 收缩 (促稀疏, 小数据慎用)


@dataclass
class FitResult:
    """BT 模型拟合结果."""

    # 后验样本: (n_samples × n_features) 的权重样本
    weight_samples: list[list[float]]
    feature_names: list[str]
    prior: PriorSpec
    n_train: int
    # 收敛诊断
    r_hat: float
    ess: float
    divergences: int
    converged: bool

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_samples(self) -> int:
        return len(self.weight_samples)

    def posterior_mean(self) -> list[float]:
        """后验均值 (logits 空间)."""
        arr = np.array(self.weight_samples)
        return arr.mean(axis=0).tolist()

    def posterior_ci(self, ci: float = 0.95) -> list[tuple[float, float]]:
        """每个特征的 95% 置信区间."""
        arr = np.array(self.weight_samples)
        alpha = (1 - ci) / 2
        lo = np.quantile(arr, alpha, axis=0)
        hi = np.quantile(arr, 1 - alpha, axis=0)
        return list(zip(lo.tolist(), hi.tolist()))

    def posterior_std(self) -> list[float]:
        arr = np.array(self.weight_samples)
        return arr.std(axis=0).tolist()

    def normalized_weights_on_simplex(self) -> list[float]:
        """把后验均值权重归一化到单纯形 (Σ=1, 全正), 便于和手工权重对比.

        注意: BT 权重的绝对值无意义, 只有相对比例有意义.
        归一化后可与 ProfileWeights 直接比较 (都落在单纯形).
        负权重 → 先 softmax-like 处理: w' = |w|, 再归一化.
        """
        mean = np.array(self.posterior_mean())
        # BT 权重可能有负值; 取绝对值后归一化 (保留相对重要性)
        abs_w = np.abs(mean)
        s = abs_w.sum()
        if s == 0:
            return [1.0 / len(mean)] * len(mean)
        return (abs_w / s).tolist()


# ── 权重 → logits 转换 (供先验用) ──────────────────────────


def weights_to_logits(weights: list[float], eps: float = 1e-3) -> list[float]:
    """把单纯形上的权重 (Σ=1, 全正) 转 logits 空间.

    logits = log(w / w_ref), w_ref = mean(w).
    这样等权重 (1/k) → logits = 0; 大权重 → 正 logits.

    注意: BT 模型里 logits 的**绝对值**无意义, 只有相对差异和符号有意义.
    先验锚定的是"哪些维度相对重要", 而非绝对数值.
    """
    w = np.clip(np.array(weights), eps, None)
    w = w / w.sum()  # 确保归一化
    log_w = np.log(w)
    return (log_w - log_w.mean()).tolist()


# ── NumPyro 模型定义 ────────────────────────────────────────


def bt_model_factory(feature_dim: int, prior: PriorSpec):
    """构造 NumPyro BT 模型函数.

    返回一个 model(diff_features, winner=None) 函数, 供 NUTS 采样.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    prior_mean = jnp.array(prior.mean_logits)
    prior_sigma = float(prior.sigma)

    def model(diff_features, winner=None):
        """
        Args:
            diff_features: (N, k) A vs B 的特征差 (潜因子 + 交叉项)
            winner: (N,) 1 if A wins, 0 if B wins; None=预测
        """
        if prior.use_horseshoe:
            # Horseshoe 收缩: 鼓励小权重趋于 0 (促稀疏)
            # 全局收缩参数
            tau = numpyro.sample("tau", dist.HalfCauchy(0.1))
            # 局部收缩参数
            lam = numpyro.sample("lam", dist.HalfCauchy(1.0).expand([feature_dim]).to_event(1))
            # 缩放后的标准差
            sigma_eff = prior_sigma * lam * tau / jnp.sqrt(1 + (lam * tau) ** 2)
            w_logits = numpyro.sample(
                "w_logits",
                dist.Normal(prior_mean, sigma_eff + 1e-6).to_event(1),
            )
        else:
            w_logits = numpyro.sample(
                "w_logits",
                dist.Normal(prior_mean, prior_sigma).to_event(1),
            )

        # 线性预测: logits = X · w
        logits = jnp.dot(diff_features, w_logits)
        numpyro.sample("winner", dist.Bernoulli(logits=logits), obs=winner)

    return model


def fit(
    features: list[list[float]],
    labels: list[int],
    prior: PriorSpec,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
    seed: int = 0,
    feature_names: Optional[list[str]] = None,
) -> FitResult:
    """拟合 BT 模型, 返回后验样本 + 收敛诊断.

    Args:
        features: (N × k) 特征矩阵
        labels: (N,) 0/1 标签
        prior: 先验规格 (mean_logits 长度需 = k)
        num_warmup: NUTS 预热步数
        num_samples: 每条链的采样数
        num_chains: 链数 (>=2 才能算 R-hat)

    Returns:
        FitResult
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    n, k = X.shape

    if len(prior.mean_logits) != k:
        raise ValueError(
            f"先验维度 {len(prior.mean_logits)} ≠ 特征维度 {k}. "
            f"先验需对每个潜因子/交叉项都有 logits."
        )

    model = bt_model_factory(k, prior)

    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples,
        num_chains=num_chains, progress_bar=False,
    )
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, diff_features=jnp.array(X), winner=jnp.array(y))

    # 提取后验样本
    samples = mcmc.get_samples()
    w_samples = np.array(samples["w_logits"])  # (num_chains * num_samples, k)

    # 收敛诊断
    try:
        from numpyro.diagnostics import summary
        # get_samples(group_by_chain=True) 保留链结构, 供 R-hat/ESS 诊断
        samples_by_chain = mcmc.get_samples(group_by_chain=True)
        w_chains = np.array(samples_by_chain["w_logits"])  # (num_chains, num_samples, k)
        # summary 返回 {site_name: {stat: scalar_or_vector}}; site 名可能被改写
        summ = summary(w_chains, group_by_chain=True)
        # 取第一个 (也是唯一) site 的统计
        first_site = next(iter(summ.values()))
        r_hat_raw = np.atleast_1d(first_site["r_hat"])
        ess_raw = np.atleast_1d(first_site["n_eff"])
        # R-hat 取最大 (最差), ESS 取最小 (最差); 过滤 nan
        valid_r = r_hat_raw[np.isfinite(r_hat_raw)]
        valid_e = ess_raw[np.isfinite(ess_raw)]
        r_hat = float(valid_r.max()) if len(valid_r) > 0 else float("nan")
        ess = float(valid_e.min()) if len(valid_e) > 0 else 0.0
    except Exception:  # noqa: BLE001
        # 诊断失败时不阻塞, 但标记未收敛
        r_hat = float("nan")
        ess = 0.0

    # divergences 计数
    extra = mcmc.get_extra_fields()
    divergences = 0
    try:
        if hasattr(extra, "get"):
            div = extra.get("diverging")
            if div is not None:
                divergences = int(np.sum(np.array(div)))
    except Exception:  # noqa: BLE001
        pass

    converged = (
        not np.isnan(r_hat) and r_hat < 1.01
        and ess > 400 / num_chains  # 每链至少 200
        and divergences < num_samples * num_chains * 0.01  # < 1% 发散
    )

    names = feature_names or [f"f{i}" for i in range(k)]

    return FitResult(
        weight_samples=w_samples.tolist(),
        feature_names=names,
        prior=prior,
        n_train=n,
        r_hat=round(r_hat, 4) if not np.isnan(r_hat) else r_hat,
        ess=round(ess, 1),
        divergences=divergences,
        converged=converged,
    )


# ── 预测 (用于评估) ─────────────────────────────────────────


def predict_proba(fit_result: FitResult, features: list[list[float]]) -> list[float]:
    """用后验样本预测 P(A wins), 返回每个样本的后验均值预测概率.

    这是 fully Bayesian 预测: 对每个后验样本算 P, 再取平均 (自动边缘化模型不确定性).
    """
    import jax.numpy as jnp

    w_samples = np.array(fit_result.weight_samples)  # (S, k)
    X = np.array(features)  # (N, k)
    # logits = X @ w^T  → (N, S)
    logits = X @ w_samples.T
    # sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.mean(axis=1).tolist()


def predict_with_fixed_weights(
    weights: list[float], features: list[list[float]]
) -> list[float]:
    """用固定权重 (如手工权重) 预测 P(A wins), 供 head-to-head 对比.

    weights 在 logits 空间; 若是单纯形权重, 调用方应先用 weights_to_logits 转换.
    """
    import numpy as np

    w = np.array(weights)
    X = np.array(features)
    logits = X @ w
    return (1.0 / (1.0 + np.exp(-logits))).tolist()
