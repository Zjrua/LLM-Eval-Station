"""加权评分引擎：维度归一化 + 缺失数据处理 + 最终加权分计算."""

from dataclasses import dataclass
from typing import Optional

from .models import Dimension, ModelScores, ProfileWeights, ScoredModel


@dataclass
class ScoringResult:
    """单个模型在单个 Profile 下的打分结果."""
    scored: ScoredModel
    skipped_dimensions: list[Dimension]
    normalized_total_weight: float

    @property
    def has_missing_data(self) -> bool:
        return len(self.skipped_dimensions) > 0


def normalize_dimension_score(raw_score: float) -> float:
    """将原始分数（0-100 或其他范围）归一化到 0-100 区间.

    处理特殊情况：
    - Arena Elo (通常 800-2000+) → 线性映射到 0-100
    - Codeforces Rating → 同上
    """
    if raw_score > 100:
        # 推测是 Elo/Rating 类分数，简单线性截断
        return min(raw_score / 20, 100.0)  # 2000 Elo → 100
    return max(0.0, min(raw_score, 100.0))


def compute_weighted_score(
    model_scores: ModelScores,
    profile_weights: ProfileWeights,
    *,
    skip_missing: bool = True,
    coverage_threshold: float = 0.3,
) -> ScoringResult:
    """计算模型在指定 Profile 下的加权总分.

    Args:
        model_scores: 模型在各维度的得分
        profile_weights: Profile 的维度权重
        skip_missing: True=缺失维度跳过并归一化剩余权重；False=缺失维度计 0 分
        coverage_threshold: 最低数据覆盖率，低于阈值时标记为「数据不足」

    Returns:
        ScoringResult 包含加权总分、详细维度得分和覆盖信息

    Raises:
        ValueError: 数据覆盖率低于阈值且 skip_missing=True 但无可用维度
    """
    available_dims = model_scores.available_dimensions
    coverage = model_scores.coverage

    if coverage < coverage_threshold:
        # 数据严重不足，返回 0 分标记
        return ScoringResult(
            scored=ScoredModel(
                model=model_scores.model,
                profile=profile_weights.profile,
                dimension_scores=dict(model_scores.scores),
                weighted_score=0.0,
                coverage=coverage,
                data_quality="❌ 数据不足",
            ),
            skipped_dimensions=model_scores.missing_dimensions,
            normalized_total_weight=0.0,
        )

    if skip_missing and available_dims:
        norm_weights = profile_weights.normalized_weights(available_dims)
    else:
        norm_weights = dict(profile_weights.weights)

    weighted_sum = 0.0
    total_weight = 0.0
    skipped: list[Dimension] = []

    for dim in Dimension:
        score = model_scores.scores.get(dim)
        if score is None:
            skipped.append(dim)
            continue
        w = norm_weights.get(dim, profile_weights.weights.get(dim, 0))
        weighted_sum += normalize_dimension_score(score) * w
        total_weight += w

    if total_weight == 0:
        weighted_score = 0.0
    else:
        weighted_score = weighted_sum / total_weight

    # 数据质量评级
    if coverage >= 0.9:
        quality = "⭐⭐⭐ 完整"
    elif coverage >= 0.6:
        quality = "⭐⭐ 部分完整"
    elif coverage >= coverage_threshold:
        quality = f"⭐ 稀疏 ({coverage:.0%})"
    else:
        quality = "❌ 数据不足"

    scored = ScoredModel(
        model=model_scores.model,
        profile=profile_weights.profile,
        dimension_scores=dict(model_scores.scores),
        weighted_score=round(weighted_score, 1),
        coverage=coverage,
        data_quality=quality,
    )

    return ScoringResult(
        scored=scored,
        skipped_dimensions=skipped,
        normalized_total_weight=round(total_weight, 3),
    )


def rank_models(
    model_scores_list: list[ModelScores],
    profile_weights: ProfileWeights,
    **kwargs,
) -> list[ScoringResult]:
    """对多个模型在指定 Profile 下评分并排名.

    Returns:
        按 weighted_score 降序排列的结果列表
    """
    results = [
        compute_weighted_score(ms, profile_weights, **kwargs)
        for ms in model_scores_list
    ]
    results.sort(key=lambda r: r.scored.weighted_score, reverse=True)
    return results


def cross_profile_ranking(
    model_scores_list: list[ModelScores],
    profiles: list[ProfileWeights],
    **kwargs,
) -> dict[str, list[ScoringResult]]:
    """多模型 × 多 Profile 的交叉评分矩阵.

    Returns:
        {profile_name: [ScoringResult, ...]}
    """
    matrix: dict[str, list[ScoringResult]] = {}
    for pw in profiles:
        results = rank_models(model_scores_list, pw, **kwargs)
        matrix[pw.profile.value] = results
    return matrix
