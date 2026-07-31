"""Profile 软标注层: 给每条 Arena 对话打 6 个 profile 的相关度评分.

设计要点 (对应"Arena 分布 ≠ Profile 分布"陷阱):
    - 不做硬分桶 (6 选 1), 而是输出 6 个 0-1 相关度 → 一条对话可属多 profile
    - 抽象成 Protocol, 默认实现 LLMAnnotator, 也提供 RuleBasedAnnotator (免费, 测试用)
    - 质量校验: 随机抽样人工复核, 计算 Cohen's κ

为什么软标注:
    Arena 用户问的"写个 Python 函数统计词频"既像 Coder (写代码) 也像 Writer (中文表述).
    硬分桶会丢失这种重叠; 软标注让下游 filter_by_profile(threshold) 自己决定纳入门槛.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Protocol

from ..models import Profile
from .data import PairwiseSample

# ── Profile 描述 (供 LLM 标注用) ───────────────────────────

PROFILE_DESCRIPTIONS: dict[Profile, str] = {
    Profile.CODER: "编程开发: 代码生成、调试、测试编写、API 调用、技术实现",
    Profile.PLANNER: "任务规划: 任务分解、依赖分析、架构设计、方案选型",
    Profile.WRITER: "文档撰写: 写作、翻译、内容创作、文案、文档生成",
    Profile.RESEARCHER: "信息调研: 信息搜索、数据收集、文献分析、事实核查",
    Profile.REVIEWER: "质量审查: 代码审查、错误发现、合规检查、fact-check",
    Profile.DOUBTER: "质疑审视: 假设质疑、漏洞发现、压力测试、风险评估",
}

ALL_PROFILE_NAMES = [p.value for p in Profile]


# ── 标注器接口 ──────────────────────────────────────────────


class ProfileAnnotator(Protocol):
    """给一段对话文本打 profile 相关度评分的标注器."""

    def annotate(self, task_text: str) -> dict[str, float]:
        """返回 {profile_name: 0-1 相关度}, 6 个 profile 都要有."""
        ...


# ── 规则标注器 (免费, 测试/冷启动用) ───────────────────────

# 关键词 → profile 的启发式映射. 召回低但精确度高, 适合做 κ 校验的下界.
_PROFILE_KEYWORDS: dict[Profile, list[str]] = {
    Profile.CODER: [
        "代码", "python", "java", "javascript", "函数", "bug", "debug",
        "编程", "实现", "算法", "code", "function", "class", "api",
        "sql", "react", "vue", "typescript", "go ", "rust",
    ],
    Profile.PLANNER: [
        "计划", "规划", "方案", "架构", "分解", "步骤", "路线图",
        "项目管理", "schedule", "plan", "roadmap", "拆解",
    ],
    Profile.WRITER: [
        "写", "作文", "文章", "文案", "翻译", "润色", "摘要", "总结",
        "故事", "诗", "邮件", "周报", "write", "translate", "essay",
    ],
    Profile.RESEARCHER: [
        "搜索", "调研", "查", "文献", "论文", "数据", "统计", "分析",
        "对比", "research", "search", "survey", "compare",
    ],
    Profile.REVIEWER: [
        "审查", "review", "检查", "审阅", "code review", "pr ", "合并请求",
        "测试覆盖", "lint", "规范",
    ],
    Profile.DOUBTER: [
        "对吗", "真的吗", "质疑", "漏洞", "风险", "问题", "反驳",
        "批判", "万一", "edge case", "边界", "stress test",
    ],
}


@dataclass
class RuleBasedAnnotator:
    """基于关键词的规则标注器. 无成本, 但召回低.

    用途:
        1. 测试 (不依赖外部 LLM API)
        2. 作为 LLMAnnotator 的下界基线, 校验 LLM 标注质量
    """

    intensity: float = 0.7  # 命中关键词时给的相关度

    def annotate(self, task_text: str) -> dict[str, float]:
        text_lower = task_text.lower()
        scores = {p.value: 0.0 for p in Profile}
        for profile, kws in _PROFILE_KEYWORDS.items():
            hits = sum(1 for kw in kws if kw.lower() in text_lower)
            if hits > 0:
                # 多关键词命中递增相关度, 但封顶
                scores[profile.value] = min(1.0, self.intensity + 0.1 * (hits - 1))
        # 若全 0, 给一个弱的 general 分 (避免 profile_scores 全空)
        if max(scores.values()) == 0:
            scores[Profile.WRITER.value] = 0.3  # 兜底: Arena 多为写作类
        return scores


# ── LLM 标注器 (生产用) ────────────────────────────────────

_ANNOTATE_PROMPT_TEMPLATE = """你是一个对话场景分类器. 判断下面的用户对话最符合哪些 AI Agent 工作场景, 给每个场景输出 0-1 的相关度评分 (可多个高分, 也可全低分).

场景定义:
{profile_defs}

对话内容:
---
{task_text}
---

请只输出 JSON, 不要其他内容, 格式:
{{"coder": 0.0, "planner": 0.0, "writer": 0.0, "researcher": 0.0, "reviewer": 0.0, "doubter": 0.0}}
"""


@dataclass
class LLMAnnotator:
    """调用 LLM API 做软标注的生产标注器.

    实现策略:
        - 默认期望 OpenAI 兼容接口 (OpenAI / DeepSeek / GLM / 本地 oMLX)
        - 抽象出 _call_llm, 子类或 patch 可换不同 provider
        - temperature=0 保证可复现
        - 解析失败时回退到 RuleBasedAnnotator

    成本估算:
        GPT-4o-mini: ~$0.15 / 1M input tokens; 33 万条 × ~300 tokens ≈ $15
        Claude Haiku: 类似量级
    """

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_chars: int = 1500  # 截断对话, 控成本

    def annotate(self, task_text: str) -> dict[str, float]:
        text = task_text[: self.max_chars]
        profile_defs = "\n".join(
            f"- {p.value}: {desc}" for p, desc in PROFILE_DESCRIPTIONS.items()
        )
        prompt = _ANNOTATE_PROMPT_TEMPLATE.format(
            profile_defs=profile_defs, task_text=text
        )

        try:
            raw = self._call_llm(prompt)
            scores = self._parse_response(raw)
            # 校验: 6 个 key 都在, 值在 [0,1]
            normalized = {}
            for name in ALL_PROFILE_NAMES:
                v = scores.get(name, 0.0)
                normalized[name] = max(0.0, min(1.0, float(v)))
            return normalized
        except Exception:  # noqa: BLE001
            # LLM 失败时回退到规则标注 (保证 pipeline 不中断)
            return RuleBasedAnnotator().annotate(task_text)

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM, 返回原始响应文本. 子类可 override 换 provider."""
        import httpx

        resp = httpx.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 200,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_response(raw: str) -> dict[str, float]:
        """解析 LLM 输出的 JSON. 容忍 markdown code fence."""
        import re

        text = raw.strip()
        # 去可能的 ```json ... ``` 包裹
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)


# ── 批量标注 ────────────────────────────────────────────────


def annotate_samples(
    samples: list[PairwiseSample],
    annotator: ProfileAnnotator,
    *,
    skip_annotated: bool = True,
) -> list[PairwiseSample]:
    """批量给样本打 profile 软标注 (原地修改 profile_scores).

    Args:
        samples: 待标注样本
        annotator: 标注器实例
        skip_annotated: 已有非空 profile_scores 的样本是否跳过 (省成本)
    """
    for s in samples:
        if skip_annotated and s.profile_scores:
            continue
        s.profile_scores = annotator.annotate(s.task_text)
    return samples


# ── 质量校验: Cohen's κ ─────────────────────────────────────


@dataclass
class AnnotationQualityReport:
    """标注质量报告."""

    n_checked: int
    # 把软标注硬分桶 (取 argmax profile) 后, 与人工标签的一致性
    cohen_kappa: float
    agreement_rate: float
    per_profile_kappa: dict[str, float]
    verdict: str  # "good" | "acceptable" | "poor"

    def is_usable(self, min_kappa: float = 0.6) -> bool:
        return self.cohen_kappa >= min_kappa


def assess_annotation_quality(
    samples: list[PairwiseSample],
    human_labels: list[str],
    *,
    n_check: int = 200,
    seed: int = 42,
) -> AnnotationQualityReport:
    """抽样对比自动标注与人工标签, 计算 Cohen's κ.

    Args:
        samples: 已自动标注的样本 (profile_scores 非空)
        human_labels: 人工硬标签列表, 长度应 >= n_check, 元素是 profile.value
            (人工从 6 选 1; 我们把自动标注也 argmax 成单标签对比)
        n_check: 抽样数

    Returns:
        AnnotationQualityReport

    Note:
        这是"软标注 vs 硬标签"的近似评估. 软标注本身允许多 profile 高分,
        但 κ 需要离散标签, 故取 argmax. 这是质量校验的下界 (真实软标注质量 >= 此报告).
    """
    rng = random.Random(seed)
    annotated = [s for s in samples if s.profile_scores]
    n = min(n_check, len(annotated), len(human_labels))
    if n == 0:
        return AnnotationQualityReport(
            n_checked=0, cohen_kappa=0.0, agreement_rate=0.0,
            per_profile_kappa={}, verdict="no_data",
        )

    indices = rng.sample(range(len(annotated)), n)
    auto_labels = [_argmax_profile(annotated[i].profile_scores) for i in indices]
    human_subset = [human_labels[i] for i in range(n)]

    kappa = _cohen_kappa(auto_labels, human_subset, ALL_PROFILE_NAMES)
    agreement = sum(a == h for a, h in zip(auto_labels, human_subset)) / n

    per_profile: dict[str, float] = {}
    for name in ALL_PROFILE_NAMES:
        auto_bin = [1 if a == name else 0 for a in auto_labels]
        human_bin = [1 if h == name else 0 for h in human_subset]
        per_profile[name] = _cohen_kappa_bin(auto_bin, human_bin)

    if kappa >= 0.6:
        verdict = "good"
    elif kappa >= 0.4:
        verdict = "acceptable"
    else:
        verdict = "poor"

    return AnnotationQualityReport(
        n_checked=n, cohen_kappa=round(kappa, 3), agreement_rate=round(agreement, 3),
        per_profile_kappa={k: round(v, 3) for k, v in per_profile.items()},
        verdict=verdict,
    )


def _argmax_profile(scores: dict[str, float]) -> str:
    """取相关度最高的 profile name (硬分桶, 用于 κ 评估)."""
    if not scores:
        return "writer"  # 兜底
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _cohen_kappa(a: list[str], b: list[str], categories: list[str]) -> float:
    """多分类 Cohen's κ."""
    import numpy as np

    n = len(a)
    if n == 0:
        return 0.0
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # 混淆矩阵
    matrix = np.zeros((k, k))
    for ai, bi in zip(a, b):
        if ai in cat_idx and bi in cat_idx:
            matrix[cat_idx[ai], cat_idx[bi]] += 1

    po = np.trace(matrix) / n  # 观察一致率
    row_marg = matrix.sum(axis=1) / n
    col_marg = matrix.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))  # 期望一致率

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _cohen_kappa_bin(a: list[int], b: list[int]) -> float:
    """二分类 Cohen's κ."""
    import numpy as np

    n = len(a)
    if n == 0:
        return 0.0
    a_arr = np.array(a)
    b_arr = np.array(b)
    po = float(np.mean(a_arr == b_arr))
    pa = float(a_arr.mean())
    pb = float(b_arr.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)
