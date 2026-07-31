"""数据层: Arena 下载 + 合成数据 + 统一 PairwiseSample schema.

三种数据源走同一接口, 下游 (features/model/evaluate) 不感知来源.

设计要点:
    - 模型维度分 (model_scores) 单独管理, 与 pairwise 样本解耦
    - profile_scores 是软标注 (0-1 相关度), 不是硬分桶; 一条对话可属多 profile
    - winner 只接受 "a"/"b"/"tie"; tie 在训练时丢弃, 评估时各计 0.5
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ..models import Dimension, Profile

# ── 统一 schema ──────────────────────────────────────────────


@dataclass
class PairwiseSample:
    """一条 A/B 偏好比较样本."""

    model_a: str  # 模型名 (需能在 ModelScoreTable 里查到维度分)
    model_b: str
    winner: str  # "a" | "b" | "tie"
    profile_scores: dict[str, float] = field(default_factory=dict)
    # 软标注: {"coder": 0.8, "reviewer": 0.6, ...}; 空表示未标注
    source: str = "unknown"  # "arena" | "synthetic" | "manual"
    task_text: str = ""  # 原始对话/任务描述, 标注与审计用
    sample_id: Optional[str] = None  # Arena 原始 id, 用于去重/追溯

    def is_decisive(self) -> bool:
        """是否有明确胜负 (训练用)."""
        return self.winner in ("a", "b")

    def profile_relevance(self, profile: Profile | str) -> float:
        """对该 profile 的相关度 (未标注返回 0)."""
        key = profile.value if isinstance(profile, Profile) else profile
        return self.profile_scores.get(key, 0.0)


@dataclass
class ModelScoreTable:
    """模型 × 维度分表. 替代旧的 data/model_scores.csv, 统一 harness 来源."""

    # {model_name: {Dimension: score}}, 缺失维度不存在 (而非 None)
    scores: dict[str, dict[Dimension, float]] = field(default_factory=dict)
    source: str = "unknown"  # "artificial_analysis" | "benchlm" | "manual"
    harness_version: str = ""  # 评测 harness 版本, 关系到可比性

    def has(self, model: str) -> bool:
        return model in self.scores

    def dims_of(self, model: str) -> list[Dimension]:
        return list(self.scores.get(model, {}).keys())

    def score(self, model: str, dim: Dimension) -> Optional[float]:
        return self.scores.get(model, {}).get(dim)

    def models(self) -> list[str]:
        return list(self.scores.keys())

    def common_dimensions(self, *models: str) -> list[Dimension]:
        """多个模型都有的维度 (用于公平比较)."""
        if not models:
            return list(Dimension)
        sets = [set(self.scores.get(m, {}).keys()) for m in models]
        common = set.intersection(*sets)
        return [d for d in Dimension if d in common]


# ── Profile 过滤 ─────────────────────────────────────────────


def filter_by_profile(
    samples: list[PairwiseSample],
    profile: Profile | str,
    threshold: float = 0.4,
) -> list[PairwiseSample]:
    """过滤出与指定 profile 相关的样本 (软标注 >= threshold).

    阈值 0.4 的含义: 一条对话若被判定与该 profile 有 40%+ 相关度, 即纳入.
    软标注允许多 profile 重叠, 故同一样本可能进多个 profile 的训练集.
    """
    return [s for s in samples if s.profile_relevance(profile) >= threshold]


def filter_covered_models(
    samples: list[PairwiseSample],
    table: ModelScoreTable,
) -> list[PairwiseSample]:
    """只保留两端模型都在维度分表里的样本.

    Arena 里有几百个模型, 但有公开维度分的只有几十个. 这一过滤会大幅缩减样本量,
    是预期行为 (我们只比较能力可量化的模型).
    """
    return [s for s in samples if table.has(s.model_a) and table.has(s.model_b)]


def train_test_split(
    samples: list[PairwiseSample],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PairwiseSample], list[PairwiseSample]]:
    """分层无关的简单随机划分; 只用 decisive 样本划分, tie 全弃.

    返回 (train, test), 两者都只含 decisive 样本.
    """
    rng = random.Random(seed)
    decisive = [s for s in samples if s.is_decisive()]
    rng.shuffle(decisive)
    n_test = int(len(decisive) * test_ratio)
    test = decisive[:n_test]
    train = decisive[n_test:]
    return train, test


# ── 合成数据生成 (sanity check 用) ──────────────────────────


def generate_synthetic_data(
    n: int = 1000,
    seed: int = 42,
    *,
    n_models: int = 12,
    noise: float = 0.3,
    profile: Profile = Profile.CODER,
    ground_truth_weights: Optional[dict[Dimension, float]] = None,
) -> tuple[list[PairwiseSample], ModelScoreTable, dict[Dimension, float]]:
    """生成合成 pairwise 数据, 已知 ground truth 权重, 用于验证 pipeline.

    流程:
        1. 随机生成 n_models 个模型的 9 维分数 (50-95 区间, 带相关性)
        2. 用 ground_truth_weights 计算每个模型的 "true 加权分"
        3. 随机配对, 用 BT 模型 + noise 生成 winner
        4. 软标注: 给定 profile 的样本标 1.0, 其他 profile 标 0.1-0.3

    Returns:
        (samples, score_table, ground_truth_weights)
    """
    from ..profiles import get_profile_weights

    rng = random.Random(seed)

    # ground truth: 默认用真实手工权重 (我们想验证"能不能学回手工权重")
    if ground_truth_weights is None:
        gt = dict(get_profile_weights(profile).weights)
    else:
        gt = dict(ground_truth_weights)

    # 1. 生成模型维度分 (带相关性, 模拟 D1/D8 共线)
    model_names = [f"synthetic-model-{i}" for i in range(n_models)]
    scores: dict[str, dict[Dimension, float]] = {}

    # 用一个潜因子生成相关分数: factor "语言" 驱动 D1/D8, "推理" 驱动 D3/D4
    for name in model_names:
        lang_factor = rng.uniform(0.4, 0.9)  # 语言能力潜因子
        reason_factor = rng.uniform(0.4, 0.9)  # 推理能力潜因子
        agent_factor = rng.uniform(0.4, 0.9)  # Agent 能力潜因子
        general = rng.uniform(0.4, 0.9)

        d_scores = {}
        # D1 综合, D8 中文 共享 lang_factor + 独立噪声
        d_scores[Dimension.D1] = _clip_score(lang_factor * 60 + rng.gauss(0, 8) + 30)
        d_scores[Dimension.D8] = _clip_score(lang_factor * 55 + rng.gauss(0, 10) + 30)
        # D3 数学, D4 科学 共享 reason_factor
        d_scores[Dimension.D3] = _clip_score(reason_factor * 65 + rng.gauss(0, 8) + 25)
        d_scores[Dimension.D4] = _clip_score(reason_factor * 60 + rng.gauss(0, 9) + 28)
        # D7 工具, D2 编程 共享 agent_factor
        d_scores[Dimension.D2] = _clip_score(agent_factor * 60 + rng.gauss(0, 9) + 28)
        d_scores[Dimension.D7] = _clip_score(agent_factor * 55 + rng.gauss(0, 10) + 30)
        # 其余维度独立
        d_scores[Dimension.D5] = _clip_score(general * 50 + rng.gauss(0, 10) + 30)
        d_scores[Dimension.D6] = _clip_score(general * 55 + rng.gauss(0, 9) + 30)
        d_scores[Dimension.D9] = _clip_score(general * 50 + rng.gauss(0, 10) + 32)
        scores[name] = d_scores

    table = ModelScoreTable(scores=scores, source="synthetic", harness_version="synthetic-v1")

    # 2. 计算 true 加权分
    def true_score(m: str) -> float:
        return sum(scores[m][d] * gt[d] for d in Dimension)

    # 3. 生成 pairwise, BT 模型: P(A beats B) = sigmoid((s_A - s_B) * scale)
    samples: list[PairwiseSample] = []
    scale = 0.15  # 控制区分度
    for i in range(n):
        a, b = rng.sample(model_names, 2)
        diff = true_score(a) - true_score(b)
        p_a_wins = _sigmoid(diff * scale)
        # 加噪声: 防止学回精确 ground truth, 测试鲁棒性
        p_noisy = _clip01(p_a_wins + rng.gauss(0, noise * 0.15))
        winner = "a" if rng.random() < p_noisy else "b"

        # 软标注: 目标 profile 高分, 其他 profile 低分
        pscores = {p.value: (0.9 if p == profile else rng.uniform(0.05, 0.25))
                   for p in Profile}
        samples.append(PairwiseSample(
            model_a=a, model_b=b, winner=winner,
            profile_scores=pscores, source="synthetic",
            task_text=f"synthetic task #{i}", sample_id=f"syn-{i}",
        ))

    return samples, table, gt


# ── Arena 数据下载与解析 ────────────────────────────────────

ARENA_DATASET_ID = "lmsys/chatbot_arena_conversations"


def load_arena_data(
    cache_dir: Path | str = "data/arena_conversations",
    *,
    max_samples: Optional[int] = None,
    force_refresh: bool = False,
) -> list[dict]:
    """从 HuggingFace Hub 下载 Chatbot Arena 对话数据.

    数据集: lmsys/chatbot_arena_conversations (~33万条带 winner 标注)

    Args:
        cache_dir: 本地缓存目录
        max_samples: 最多返回多少条 (调试用); None=全部
        force_refresh: 忽略缓存强制重下

    Returns:
        原始 Arena 记录列表 (未转 PairwiseSample), 每条是 dict:
            {"model_a": ..., "model_b": ..., "winner": ...,
             "conversation_a": [...], "conversation_b": [...]}

    Note:
        实际下载需 huggingface_hub. 调用方应捕获网络错误.
        Arena 的 winner 字段为 "model_a"/"model_b"/"tie", 本函数原样返回.
    """
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Arena 数据集以 parquet/json 提供, 文件名可能随版本变; 这里尝试常见路径
    # 真实结构 (2026 实测): data/train-00000-of-00001-*.parquet
    repo_id = ARENA_DATASET_ID
    candidates = [
        ("data/train-00000-of-00001-cced8514c7ed782a.parquet", "parquet"),
        ("data/train-00000-of-00001.parquet", "parquet"),
        ("train-00000-of-00001.parquet", "parquet"),
        ("chatbot_arena_conversations.json", "json"),
    ]

    records: list[dict] = []
    last_err: Optional[Exception] = None
    for path_in_repo, fmt in candidates:
        try:
            local = hf_hub_download(
                repo_id=repo_id, filename=path_in_repo,
                repo_type="dataset", cache_dir=str(cache_dir),
                force_download=force_refresh,
            )
            records = _load_arena_file(local, fmt)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    if not records:
        raise RuntimeError(
            f"无法从 {repo_id} 下载数据 (试过 {len(candidates)} 个路径). "
            f"最后错误: {last_err}"
        )

    if max_samples is not None:
        records = records[:max_samples]
    return records


def _load_arena_file(path: str, fmt: str) -> list[dict]:
    """根据格式加载 Arena 数据文件."""
    if fmt == "parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")
    elif fmt == "json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"未知格式: {fmt}")


def arena_records_to_samples(
    records: list[dict],
    *,
    normalize_model_names: bool = True,
) -> list[PairwiseSample]:
    """把 Arena 原始记录转成 PairwiseSample (尚未软标注).

    Arena winner 值 "model_a"/"model_b"/"tie" → 标准化为 "a"/"b"/"tie".
    profile_scores 此时为空, 由 profiling.annotate_samples() 填充.
    """
    samples: list[PairwiseSample] = []
    for rec in records:
        a = rec.get("model_a", "")
        b = rec.get("model_b", "")
        winner_raw = rec.get("winner", "")
        if normalize_model_names:
            a = _normalize_model_name(a)
            b = _normalize_model_name(b)

        winner = {"model_a": "a", "model_b": "b", "tie": "tie"}.get(winner_raw, "")
        if not winner:
            continue  # 跳过异常 winner

        # 拼接对话文本用于后续标注
        conv_a = rec.get("conversation_a", [])
        conv_b = rec.get("conversation_b", [])
        task_text = _conv_to_text(conv_a) if conv_a else ""

        samples.append(PairwiseSample(
            model_a=a, model_b=b, winner=winner,
            profile_scores={}, source="arena",
            task_text=task_text,
            sample_id=rec.get("question_id") or rec.get("id"),
        ))
    return samples


def _normalize_model_name(name: str) -> str:
    """Arena 模型名标准化 (去后缀、统一大小写).

    例: "gpt-4-turbo-2024-04-09" → "gpt-4-turbo"
        "claude-3-opus-20240229" → "claude-3-opus"
    匹配维度分表时, 调用方仍需用模糊匹配兜底.
    """
    name = name.strip().lower()
    # 去日期后缀
    import re
    name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)
    name = re.sub(r"-\d{8}$", "", name)
    return name


def _conv_to_text(conv: list) -> str:
    """把 Arena 对话列表拼成纯文本 (前 2 轮, 控制长度)."""
    parts = []
    for turn in conv[:4]:  # 前 4 个 turn (通常 user/assistant 交替)
        role = turn.get("role", "")
        content = turn.get("content", "")
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)[:2000]  # 截断, 控制 LLM 标注成本


# ── 持久化 (PairwiseSample 列表) ────────────────────────────


def save_samples_jsonl(samples: list[PairwiseSample], path: Path | str) -> None:
    """把样本列表存为 JSONL (每行一个样本)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            # Dimension 枚举不能直接 JSON 序列化, profile_scores 的 key 已经是 str
            d = asdict(s)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def load_samples_jsonl(path: Path | str) -> list[PairwiseSample]:
    """从 JSONL 读回样本列表."""
    path = Path(path)
    samples: list[PairwiseSample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(PairwiseSample(**d))
    return samples


# ── 数值工具 ────────────────────────────────────────────────


def _sigmoid(x: float) -> float:
    import math
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip_score(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
