"""评估层: held-out 评估 + vs 手工权重对比 + decision rule.

核心验收标准 (对应"必须显著超越手工"门槛):
    学出权重的 held-out 预测准确率 - 手工权重准确率 >= 5pp 才上线 (USE_LEARNED)
    否则 fallback 到手工权重, 明确标注原因

诚实评估原则:
    - 每 Profile 独立评估, 不平均
    - 评估指标: accuracy + Brier score (校准度)
    - 决策透明: 每个 profile 给出 USE_LEARNED / FALLBACK_HANDCRAFT + 理由
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from ..models import Dimension, Profile
from .data import PairwiseSample, train_test_split
from .features import build_feature_matrix, FactorModel
from .model import predict_proba, predict_with_fixed_weights, weights_to_logits
from .train import ProfileLearningResult, project_weights_to_factors


@dataclass
class ProfileEval:
    """单 Profile 的评估结果."""

    profile: Profile
    n_heldout: int

    # 学出权重的表现
    heldout_accuracy_learned: float
    brier_score_learned: float

    # 手工权重的表现 (head-to-head baseline)
    heldout_accuracy_handcraft: float
    brier_score_handcraft: float

    # 关键决策指标
    lift_pp: float  # = accuracy_learned - accuracy_handcraft (百分点)
    passes_threshold: bool  # lift_pp >= 5.0
    ci_too_wide: bool  # 是否有维度 CI 宽度 > 0.15 (权重不确定)

    # 最终决策
    decision: str  # "USE_LEARNED" | "FALLBACK_HANDCRAFT"
    reason: str = ""

    def is_use_learned(self) -> bool:
        return self.decision == "USE_LEARNED"


# ── 决策规则 ────────────────────────────────────────────────

LIFT_THRESHOLD_PP = 5.0  # 5 个百分点
CI_WIDTH_THRESHOLD = 0.15  # 单维度 CI 宽度上限


def decision_rule(
    lift_pp: float,
    ci_too_wide: bool,
    *,
    converged: bool,
    threshold: float = LIFT_THRESHOLD_PP,
) -> tuple[str, str]:
    """根据评估指标决定是否使用学出的权重.

    逻辑:
        1. 模型未收敛 → FALLBACK (结果不可信)
        2. CI 过宽 → FALLBACK (权重不可识别)
        3. lift < threshold → FALLBACK (数据未带来显著增益)
        4. 否则 → USE_LEARNED

    Returns:
        (decision, reason)
    """
    if not converged:
        return "FALLBACK_HANDCRAFT", "模型未收敛, 学出权重不可信"
    if ci_too_wide:
        return "FALLBACK_HANDCRAFT", (
            f"权重置信区间过宽 (有维度 CI 宽度 > {CI_WIDTH_THRESHOLD}), "
            "共线性导致权重不可单独识别"
        )
    if lift_pp < threshold:
        return "FALLBACK_HANDCRAFT", (
            f"lift 仅 {lift_pp:.1f}pp < {threshold}pp 阈值, "
            "学出权重未显著超越手工, 手工权重已足够"
        )
    return "USE_LEARNED", (
        f"lift {lift_pp:.1f}pp ≥ {threshold}pp 阈值, "
        "学出权重显著超越手工, 采用学出权重"
    )


# ── 评估单个 Profile ────────────────────────────────────────


def evaluate_profile(
    result: ProfileLearningResult,
    test_samples: list[PairwiseSample],
    table,
    factor_model: FactorModel,
) -> ProfileEval:
    """在 held-out 上评估单 Profile.

    Args:
        result: train_profile 的结果 (含 fit_result, factor_model 等)
        test_samples: held-out 测试样本
        table: 模型维度分表
        factor_model: 拟合好的因子模型

    Returns:
        ProfileEval
    """
    import numpy as np

    profile = result.profile

    # 跳过的情况: 没训出模型
    if result.fit_result is None or result.factor_model is None:
        return ProfileEval(
            profile=profile, n_heldout=0,
            heldout_accuracy_learned=0.0, brier_score_learned=1.0,
            heldout_accuracy_handcraft=0.0, brier_score_handcraft=1.0,
            lift_pp=0.0, passes_threshold=False, ci_too_wide=False,
            decision="FALLBACK_HANDCRAFT",
            reason=result.skipped_reason or "模型未训练",
        )

    # 构造 held-out 特征
    features, labels, _ = build_feature_matrix(test_samples, table, factor_model)
    if not features:
        return ProfileEval(
            profile=profile, n_heldout=0,
            heldout_accuracy_learned=0.0, brier_score_learned=1.0,
            heldout_accuracy_handcraft=0.0, brier_score_handcraft=1.0,
            lift_pp=0.0, passes_threshold=False, ci_too_wide=False,
            decision="FALLBACK_HANDCRAFT",
            reason="held-out 样本无法构造特征",
        )

    y = np.array(labels)
    n = len(y)

    # 学出权重的预测 (fully Bayesian: 边缘化后验)
    probs_learned = np.array(predict_proba(result.fit_result, features))
    preds_learned = (probs_learned >= 0.5).astype(int)
    acc_learned = float((preds_learned == y).mean())
    brier_learned = float(np.mean((probs_learned - y) ** 2))

    # 手工权重的预测 (baseline)
    # 手工权重在原维度空间, 投影到潜因子空间作 logits
    from ..profiles import get_profile_weights

    handcraft = get_profile_weights(profile).weights
    handcraft_logits = project_weights_to_factors(
        handcraft, result.factor_model, result.feature_names,
    )
    probs_handcraft = np.array(
        predict_with_fixed_weights(handcraft_logits, features)
    )
    preds_handcraft = (probs_handcraft >= 0.5).astype(int)
    acc_handcraft = float((preds_handcraft == y).mean())
    brier_handcraft = float(np.mean((probs_handcraft - y) ** 2))

    # lift (百分点)
    lift_pp = (acc_learned - acc_handcraft) * 100.0

    # CI 宽度检查
    ci_widths = [hi - lo for lo, hi in result.dim_weights_ci.values()]
    ci_too_wide = bool(any(w > CI_WIDTH_THRESHOLD for w in ci_widths))

    converged = result.fit_result.converged
    passes = lift_pp >= LIFT_THRESHOLD_PP
    decision, reason = decision_rule(
        lift_pp, ci_too_wide, converged=converged,
    )

    return ProfileEval(
        profile=profile, n_heldout=n,
        heldout_accuracy_learned=round(acc_learned, 4),
        brier_score_learned=round(brier_learned, 4),
        heldout_accuracy_handcraft=round(acc_handcraft, 4),
        brier_score_handcraft=round(brier_handcraft, 4),
        lift_pp=round(lift_pp, 2),
        passes_threshold=passes,
        ci_too_wide=ci_too_wide,
        decision=decision,
        reason=reason,
    )


# ── 汇总报告 ────────────────────────────────────────────────


@dataclass
class LearningEvalSummary:
    """全 Profile 评估汇总."""

    profiles: dict[str, ProfileEval] = field(default_factory=dict)
    data_source: str = ""
    trained_at: str = ""

    @property
    def n_use_learned(self) -> int:
        return sum(1 for e in self.profiles.values() if e.is_use_learned())

    @property
    def n_total(self) -> int:
        return len(self.profiles)

    def to_markdown(self) -> str:
        lines = ["## 权重学习评估汇总\n"]
        lines.append(f"- 数据源: {self.data_source}")
        lines.append(f"- 训练时间: {self.trained_at}")
        lines.append(
            f"- 决策: {self.n_use_learned}/{self.n_total} profile 采用学出权重\n"
        )
        lines.append("### 各 Profile 详情\n")
        lines.append("| Profile | 学出 Acc | 手工 Acc | Lift | Brier(学) | 决策 |")
        lines.append("|---------|---------|---------|------|----------|------|")
        for name, ev in self.profiles.items():
            emoji = "✅" if ev.is_use_learned() else "↩️"
            lines.append(
                f"| {name} | {ev.heldout_accuracy_learned:.1%} | "
                f"{ev.heldout_accuracy_handcraft:.1%} | "
                f"{ev.lift_pp:+.1f}pp | {ev.brier_score_learned:.3f} | "
                f"{emoji} {ev.decision} |"
            )
        lines.append("")
        if self.n_use_learned == 0:
            lines.append(
                "> ⚠️ **无 Profile 通过 5pp 门槛**. 这是合法负面结果: "
                "Arena 公开偏好数据对你 6 个 Profile 的区分度不足, "
                "手工权重在当前数据规模下已足够。建议:\n"
                "> 1. 采集真实 Hermes 任务成功率数据 (路径 C)\n"
                "> 2. 扩大模型池 (当前维度分模型数限制信号)\n"
            )
        return "\n".join(lines)


def summarize_evals(
    evals: dict[Profile, ProfileEval],
    *,
    data_source: str = "",
    trained_at: str = "",
) -> LearningEvalSummary:
    """把 per-profile 评估汇总成报告对象."""
    from datetime import datetime

    return LearningEvalSummary(
        profiles={p.value: e for p, e in evals.items()},
        data_source=data_source,
        trained_at=trained_at or datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
