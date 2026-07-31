"""训练编排: 串联数据 → 特征 → BT 模型, 单 Profile + 全 Profile.

提供:
    train_profile()   — 单 Profile 训练
    train_all()       — 6 个 Profile 顺序训练
    sanity_check()    — 合成数据 sanity check (验证 pipeline 能学回 ground truth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models import Dimension, Profile
from ..profiles import get_profile_weights
from .data import (
    ModelScoreTable,
    PairwiseSample,
    filter_by_profile,
    filter_covered_models,
    train_test_split,
)
from .features import (
    CollinearityReport,
    CrossTermSpec,
    DEFAULT_CROSS_TERMS,
    FactorModel,
    build_feature_matrix,
    diagnose_collinearity,
    extract_latent_factors,
)
from .model import (
    FitResult,
    PriorSpec,
    fit as fit_model,
    weights_to_logits,
)


@dataclass
class ProfileLearningResult:
    """单 Profile 训练全结果."""

    profile: Profile
    # 数据
    n_samples_total: int
    n_samples_train: int
    n_samples_test: int
    # 特征
    factor_model: Optional[FactorModel]
    n_features: int
    feature_names: list[str]
    # 模型
    fit_result: Optional[FitResult]
    # 派生: 原维度等效权重 + CI (反投影自潜因子)
    dim_weights: dict[str, float]  # {D1: 0.13, ...} 归一化到单纯形
    dim_weights_ci: dict[str, tuple[float, float]]
    # 对比
    handcraft_weights: dict[str, float]
    # 诊断
    collinearity: Optional[CollinearityReport]
    skipped_reason: str = ""  # 若跳过, 说明原因

    @property
    def learned_ok(self) -> bool:
        return self.fit_result is not None and self.fit_result.converged


def train_profile(
    samples: list[PairwiseSample],
    table: ModelScoreTable,
    profile: Profile,
    *,
    profile_threshold: float = 0.4,
    test_ratio: float = 0.2,
    prior_sigma: float = 0.5,
    num_warmup: int = 500,
    num_samples: int = 1000,
    seed: int = 0,
    min_samples: int = 100,
) -> ProfileLearningResult:
    """训练单个 Profile 的 BT 权重.

    流程:
        1. 过滤: profile 相关 + 模型覆盖
        2. 共线性诊断
        3. 因子分析降维
        4. 构造特征矩阵 (含交叉项)
        5. 训练 BT 模型 (NumPyro NUTS)
        6. 反投影: 潜因子权重 → 原维度等效权重 + CI

    Args:
        min_samples: 训练集最少样本数, 不足则跳过 (返回 skipped)
    """
    # 1. 过滤
    relevant = filter_by_profile(samples, profile, profile_threshold)
    covered = filter_covered_models(relevant, table)
    if len(covered) < min_samples:
        return _skipped_result(
            profile, len(relevant), len(covered),
            f"样本不足: 相关 {len(relevant)}, 覆盖 {len(covered)} < {min_samples}",
        )

    train, test = train_test_split(covered, test_ratio=test_ratio, seed=seed)
    if len(train) < min_samples:
        return _skipped_result(
            profile, len(relevant), len(train),
            f"训练集不足: {len(train)} < {min_samples}",
        )

    # 2. 共线性诊断
    collinearity = diagnose_collinearity(table)

    # 3. 因子分析降维
    factor_model = extract_latent_factors(table)
    if factor_model is None:
        return _skipped_result(
            profile, len(relevant), len(train),
            "因子分析失败 (模型数或维度不足)",
            collinearity=collinearity,
        )

    # 4. 构造特征矩阵
    features, labels, feature_names = build_feature_matrix(
        train, table, factor_model,
    )
    if len(features) < min_samples:
        return _skipped_result(
            profile, len(relevant), len(features),
            f"有效特征样本不足: {len(features)} < {min_samples}",
            factor_model=factor_model, collinearity=collinearity,
        )

    # 5. 先验: 手工权重 → logits
    handcraft = get_profile_weights(profile).weights
    # 手工权重在原维度空间, 需先聚合到潜因子空间
    prior_logits = project_weights_to_factors(handcraft, factor_model, feature_names)
    prior = PriorSpec(
        mean_logits=prior_logits,
        sigma=prior_sigma,
        use_horseshoe=False,  # 小数据下 horseshoe 风险高, 默认关
    )

    # 6. 训练
    fit_result = fit_model(
        features, labels, prior,
        num_warmup=num_warmup, num_samples=num_samples,
        num_chains=2, seed=seed,
        feature_names=feature_names,
    )

    # 7. 反投影: 潜因子权重 → 原维度
    dim_weights, dim_ci = _backproject_weights(fit_result, factor_model, feature_names)

    return ProfileLearningResult(
        profile=profile,
        n_samples_total=len(relevant),
        n_samples_train=len(train),
        n_samples_test=len(test),
        factor_model=factor_model,
        n_features=len(feature_names),
        feature_names=feature_names,
        fit_result=fit_result,
        dim_weights=dim_weights,
        dim_weights_ci=dim_ci,
        handcraft_weights={d.name: w for d, w in handcraft.items()},
        collinearity=collinearity,
    )


def train_all(
    samples: list[PairwiseSample],
    table: ModelScoreTable,
    **kwargs,
) -> dict[Profile, ProfileLearningResult]:
    """顺序训练 6 个 Profile.

    顺序而非并行: 因子分析 + 共线性诊断对所有 profile 共享 (本可缓存),
    但当前实现每 profile 重算 (代码简单优先, 性能不是瓶颈).
    """
    results: dict[Profile, ProfileLearningResult] = {}
    for profile in Profile:
        results[profile] = train_profile(samples, table, profile, **kwargs)
    return results


# ── 合成数据 sanity check ──────────────────────────────────


@dataclass
class SanityCheckReport:
    """合成数据 sanity check 报告."""

    profile: Profile
    ground_truth_weights: dict[str, float]
    learned_dim_weights: dict[str, float]
    # 比对指标
    mae: float  # 平均绝对误差 (单纯形权重)
    correlation: float  # 学出权重 vs ground truth 的相关性
    fit_converged: bool
    passed: bool  # MAE < 0.05 且 converged

    def to_markdown(self) -> str:
        lines = [f"### Sanity Check: {self.profile.value}\n"]
        lines.append("| 维度 | Ground Truth | 学出 | 偏差 |")
        lines.append("|------|-------------|------|------|")
        for d in sorted(self.ground_truth_weights.keys()):
            gt = self.ground_truth_weights[d]
            learned = self.learned_dim_weights.get(d, 0.0)
            lines.append(f"| {d} | {gt:.3f} | {learned:.3f} | {abs(gt-learned):.3f} |")
        lines.append(f"\n- MAE: {self.mae:.4f}")
        lines.append(f"- 相关性: {self.correlation:.3f}")
        lines.append(f"- 收敛: {'✓' if self.fit_converged else '✗'}")
        lines.append(f"- 判定: {'✓ 通过' if self.passed else '✗ 未通过'}")
        return "\n".join(lines)


def sanity_check(
    *,
    n_samples: int = 2000,
    seed: int = 42,
    profile: Profile = Profile.CODER,
    prior_sigma: float = 0.8,  # sanity check 用弱先验, 让数据说话
    num_warmup: int = 300,
    num_samples: int = 800,
) -> SanityCheckReport:
    """合成数据 sanity check: 验证 pipeline 能学回 ground truth.

    流程:
        1. 用真实手工权重生成合成 pairwise 数据
        2. 跑完整 train_profile pipeline
        3. 比对学出权重与 ground truth

    通过标准: MAE < 0.05 且模型收敛 (注意: 因子分析后反投影会有信息损失,
    纯共线维度无法精确还原, 所以 MAE 阈值放宽到 0.05).
    """
    from .data import generate_synthetic_data

    samples, table, gt_weights = generate_synthetic_data(
        n=n_samples, seed=seed, profile=profile,
        ground_truth_weights=None,  # 用默认 (即真实手工权重)
    )

    result = train_profile(
        samples, table, profile,
        profile_threshold=0.5,  # 合成数据软标注明确, 用高阈值
        prior_sigma=prior_sigma,
        num_warmup=num_warmup,
        num_samples=num_samples,
        seed=seed,
        min_samples=200,
    )

    if not result.learned_ok:
        return SanityCheckReport(
            profile=profile,
            ground_truth_weights={d.name: w for d, w in gt_weights.items()},
            learned_dim_weights={},
            mae=1.0,
            correlation=0.0,
            fit_converged=False,
            passed=False,
        )

    # 计算 MAE 和相关性
    import numpy as np

    # 统一用维度名字符串排序对齐: gt_weights key 是 Dimension, result.dim_weights key 是 "D1"
    dim_order = sorted(d.name for d in gt_weights.keys())  # ['D1', 'D2', ...]
    gt_arr = np.array([gt_weights[Dimension[d]] for d in dim_order])
    learned_arr = np.array([result.dim_weights.get(d, 0.0) for d in dim_order])

    mae = float(np.mean(np.abs(gt_arr - learned_arr)))
    if gt_arr.std() > 0 and learned_arr.std() > 0:
        corr = float(np.corrcoef(gt_arr, learned_arr)[0, 1])
    else:
        corr = 0.0

    passed = mae < 0.05 and result.fit_result.converged

    return SanityCheckReport(
        profile=profile,
        ground_truth_weights={d: gt_weights[Dimension[d]] for d in dim_order},
        learned_dim_weights={d: result.dim_weights.get(d, 0.0) for d in dim_order},
        mae=round(mae, 4),
        correlation=round(corr, 3),
        fit_converged=result.fit_result.converged,
        passed=passed,
    )


# ── 内部工具 ────────────────────────────────────────────────


def _skipped_result(
    profile: Profile,
    n_relevant: int,
    n_usable: int,
    reason: str,
    *,
    factor_model: Optional[FactorModel] = None,
    collinearity: Optional[CollinearityReport] = None,
) -> ProfileLearningResult:
    """构造跳过的结果 (数据不足等情况)."""
    handcraft = get_profile_weights(profile).weights
    return ProfileLearningResult(
        profile=profile,
        n_samples_total=n_relevant,
        n_samples_train=0,
        n_samples_test=0,
        factor_model=factor_model,
        n_features=0,
        feature_names=[],
        fit_result=None,
        dim_weights={d.name: w for d, w in handcraft.items()},  # fallback 手工
        dim_weights_ci={d.name: (w, w) for d, w in handcraft.items()},
        handcraft_weights={d.name: w for d, w in handcraft.items()},
        collinearity=collinearity,
        skipped_reason=reason,
    )


def project_weights_to_factors(
    handcraft: dict[Dimension, float],
    factor_model: FactorModel,
    feature_names: list[str],
) -> list[float]:
    """把原维度权重投影到潜因子空间 (作 BT 模型先验).

    流程:
        1. 手工权重 (单纯形) → logits
        2. 用 loading_matrix 反向投影: factor_logits = loading @ dim_logits

    交叉项的先验设为 0 (我们不预设交互效应方向).

    公共函数: train.py 作先验, evaluate.py 作手工 baseline 预测时都用.
    """
    # 原维度 logits
    dim_order = [Dimension[d] for d in factor_model.dims_in_order]
    w_simplex = [handcraft.get(d, 1.0 / len(dim_order)) for d in dim_order]
    dim_logits = weights_to_logits(w_simplex)

    # 投影到潜因子: loading_matrix 是 (k × 9), dim_logits 是 (9,)
    import numpy as np
    loadings = np.array(factor_model.loading_matrix)  # (k, 9)
    factor_logits = (loadings @ np.array(dim_logits)).tolist()

    # 追加交叉项的先验 (0 = 无预设方向)
    n_cross = len(feature_names) - factor_model.n_factors
    return factor_logits + [0.0] * n_cross


def _backproject_weights(
    fit_result: FitResult,
    factor_model: FactorModel,
    feature_names: list[str],
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """把潜因子空间的后验权重反投影到原维度.

    流程:
        1. 取潜因子部分的权重 (排除交叉项)
        2. 反投影: dim_weights = loading^T @ factor_weights
        3. 取绝对值 + 归一化到单纯形
        4. 同样对每个后验样本做, 得 CI

    交叉项不反投影 (它们是独立的交互项, 已在原维度空间).
    """
    import numpy as np

    # 分离潜因子权重和交叉项权重
    n_factors = factor_model.n_factors
    w_samples = np.array(fit_result.weight_samples)  # (S, total_features)
    factor_samples = w_samples[:, :n_factors]  # (S, k)
    # 交叉项样本暂不用于反投影 (它们已在原维度空间)

    loadings = np.array(factor_model.loading_matrix)  # (k, 9)
    # 反投影每个后验样本: (S, 9) = (S, k) @ (k, 9)
    dim_samples = factor_samples @ loadings

    # 取绝对值 + 归一化 (每个样本独立归一化)
    abs_samples = np.abs(dim_samples)
    sums = abs_samples.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    norm_samples = abs_samples / sums  # (S, 9)

    dim_order = factor_model.dims_in_order
    mean_weights = norm_samples.mean(axis=0)
    ci_lo = np.quantile(norm_samples, 0.025, axis=0)
    ci_hi = np.quantile(norm_samples, 0.975, axis=0)

    # 归一化均值到 Σ=1 (均值后可能略微偏离 1)
    total = mean_weights.sum()
    if total > 0:
        mean_weights = mean_weights / total

    dim_weights = {dim_order[i]: float(mean_weights[i]) for i in range(len(dim_order))}
    dim_ci = {
        dim_order[i]: (float(ci_lo[i]), float(ci_hi[i]))
        for i in range(len(dim_order))
    }

    return dim_weights, dim_ci
