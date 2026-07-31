"""权重持久化: 序列化训练结果为 JSON, 反序列化为 ProfileWeights.

learned_weights.json 结构 (阶段 4 产物, 阶段 5 evaluate.py 消费):
    {
      "version": "0.1.0",
      "trained_at": "2026-07-26",
      "data_source": "lmsys/chatbot_arena_conversations",
      "profiles": {
        "coder": {
          "decision": "USE_LEARNED",
          "weights": {"D1": 0.13, ...},          # 归一化到单纯形
          "weights_ci": {"D1": [0.08, 0.18]},    # 95% CI
          "handcraft_weights": {"D1": 0.15, ...},
          "lift_pp": 6.2,
          "eval": {accuracy/brier/...},
          "converged": true,
          "n_train": 1200
        },
        "doubter": {
          "decision": "FALLBACK_HANDCRAFT",
          "reason": "...",
          "weights": {...},  # = handcraft (fallback 时)
          ...
        }
      }
    }

关键约束:
    - weights 必须 Σ=1.0 (ProfileWeights.__post_init__ 强制)
    - USE_LEARNED 的 weights 用学出值; FALLBACK 的 weights = handcraft
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ..models import Dimension, Profile, ProfileWeights
from ..profiles import get_profile_weights
from .evaluate import ProfileEval


@dataclass
class ProfileWeightsArtifact:
    """单 Profile 的权重产物 (可序列化)."""

    decision: str  # "USE_LEARNED" | "FALLBACK_HANDCRAFT"
    weights: dict[str, float]  # {D1: 0.13, ...} Σ=1.0
    weights_ci: dict[str, list[float]]  # {D1: [lo, hi]}
    handcraft_weights: dict[str, float]
    reason: str = ""
    # 评估元数据
    lift_pp: float = 0.0
    heldout_accuracy_learned: float = 0.0
    heldout_accuracy_handcraft: float = 0.0
    brier_score_learned: float = 0.0
    converged: bool = False
    n_train: int = 0
    n_heldout: int = 0


@dataclass
class LearnedWeightsFile:
    """learned_weights.json 的完整结构."""

    version: str = "0.1.0"
    trained_at: str = ""
    data_source: str = ""
    profiles: dict[str, ProfileWeightsArtifact] = field(default_factory=dict)

    @property
    def n_use_learned(self) -> int:
        return sum(1 for a in self.profiles.values() if a.decision == "USE_LEARNED")


def build_artifact_from_result(
    learning_result,  # ProfileLearningResult
    eval_result: ProfileEval,
) -> ProfileWeightsArtifact:
    """从训练结果 + 评估结果构造可序列化的产物.

    决定 weights 字段用学出值还是手工值:
        - USE_LEARNED → 用学出的 dim_weights
        - FALLBACK → 用 handcraft_weights
    """
    if eval_result.is_use_learned():
        weights = dict(learning_result.dim_weights)
        weights_ci = {
            d: [lo, hi] for d, (lo, hi) in learning_result.dim_weights_ci.items()
        }
    else:
        # fallback: 用手工权重, CI 退化为点估计
        weights = dict(learning_result.handcraft_weights)
        weights_ci = {d: [w, w] for d, w in weights.items()}

    # 兜底: 若 weights 不全 9 维 (可能因因子分析维度缺失), 用手工补齐
    for d in Dimension:
        if d.name not in weights:
            weights[d.name] = learning_result.handcraft_weights.get(d.name, 1.0 / 9)
            weights_ci[d.name] = [weights[d.name], weights[d.name]]

    # 确保归一化 (浮点累积误差修正)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return ProfileWeightsArtifact(
        decision=eval_result.decision,
        weights=weights,
        weights_ci=weights_ci,
        handcraft_weights=dict(learning_result.handcraft_weights),
        reason=eval_result.reason,
        lift_pp=eval_result.lift_pp,
        heldout_accuracy_learned=eval_result.heldout_accuracy_learned,
        heldout_accuracy_handcraft=eval_result.heldout_accuracy_handcraft,
        brier_score_learned=eval_result.brier_score_learned,
        converged=(learning_result.fit_result.converged
                   if learning_result.fit_result is not None else False),
        n_train=learning_result.n_samples_train,
        n_heldout=eval_result.n_heldout,
    )


def save_learned_weights(
    artifacts: dict[Profile, ProfileWeightsArtifact],
    path: Path | str,
    *,
    data_source: str = "",
    trained_at: str = "",
) -> None:
    """保存学出的权重到 JSON.

    Args:
        artifacts: per-profile 产物
        path: 输出路径
        data_source: 数据来源描述
        trained_at: 训练时间
    """
    from datetime import datetime

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_obj = LearnedWeightsFile(
        version="0.1.0",
        trained_at=trained_at or datetime.now().strftime("%Y-%m-%d %H:%M"),
        data_source=data_source,
        profiles={p.value: a for p, a in artifacts.items()},
    )

    # dataclasses 嵌套 → dict → JSON
    data = {
        "version": file_obj.version,
        "trained_at": file_obj.trained_at,
        "data_source": file_obj.data_source,
        "profiles": {
            name: asdict(artifact) for name, artifact in file_obj.profiles.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_learned_weights(path: Path | str) -> LearnedWeightsFile:
    """从 JSON 读回权重文件."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    profiles: dict[str, ProfileWeightsArtifact] = {}
    for name, pdata in data.get("profiles", {}).items():
        profiles[name] = ProfileWeightsArtifact(**pdata)

    return LearnedWeightsFile(
        version=data.get("version", "0.1.0"),
        trained_at=data.get("trained_at", ""),
        data_source=data.get("data_source", ""),
        profiles=profiles,
    )


# ── 转成 ProfileWeights (供 scoring.py 消费) ────────────────


def to_profile_weights(
    artifact: ProfileWeightsArtifact,
    profile: Profile,
) -> ProfileWeights:
    """把单个 artifact 转成核心包的 ProfileWeights (drop-in).

    保证: weights 已归一化到 Σ=1.0 (满足 ProfileWeights.__post_init__).
    """
    # 把 str key "D1" 转回 Dimension enum
    dim_weights: dict[Dimension, float] = {}
    for dim_name, w in artifact.weights.items():
        try:
            dim_weights[Dimension[dim_name]] = w
        except KeyError:
            continue  # 忽略未知维度

    # ProfileWeights 构造时会校验 Σ=1.0 ± 0.01
    return ProfileWeights(profile=profile, weights=dim_weights)


def load_profile_weights_for_eval(
    path: Path | str,
    *,
    mode: str = "auto",
) -> dict[Profile, ProfileWeights]:
    """加载权重, 按 mode 决定每个 profile 用学出还是手工.

    Args:
        path: learned_weights.json 路径
        mode: "auto" | "learned" | "handcraft"
            - auto: 按 artifact.decision (USE_LEARNED 用学出, FALLBACK 用手工)
            - learned: 强制全用学出 (即便 decision 是 FALLBACK, 用于对比)
            - handcraft: 强制全用手工 (忽略文件)

    Returns:
        {Profile: ProfileWeights}
    """
    if mode == "handcraft":
        return {p: get_profile_weights(p) for p in Profile}

    loaded = load_learned_weights(path)
    result: dict[Profile, ProfileWeights] = {}
    for p in Profile:
        artifact = loaded.profiles.get(p.value)
        if artifact is None:
            result[p] = get_profile_weights(p)
            continue

        if mode == "learned":
            # 强制用学出权重 (artifact.weights 在 USE_LEARNED 时是学出值,
            # 在 FALLBACK 时也是 handcraft, 所以直接用)
            result[p] = to_profile_weights(artifact, p)
        else:  # auto
            result[p] = to_profile_weights(artifact, p)

    return result


def get_learned_profiles_summary(path: Path | str) -> str:
    """生成 learned_weights.json 的人类可读摘要 (供 CLI 打印)."""
    loaded = load_learned_weights(path)
    lines = [
        f"学出权重文件: {path}",
        f"  版本: {loaded.version}, 训练于 {loaded.trained_at}",
        f"  数据源: {loaded.data_source}",
        f"  决策: {loaded.n_use_learned}/{len(loaded.profiles)} profile 采用学出权重\n",
    ]
    for name, art in loaded.profiles.items():
        emoji = "✅" if art.decision == "USE_LEARNED" else "↩️"
        lines.append(
            f"  {emoji} {name}: lift {art.lift_pp:+.1f}pp, "
            f"acc {art.heldout_accuracy_learned:.1%} vs 手工 {art.heldout_accuracy_handcraft:.1%}"
        )
    return "\n".join(lines)
