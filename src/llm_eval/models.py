"""数据模型：Model, Dimension, Profile, Benchmark."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Dimension(Enum):
    """9 大能力维度."""
    D1 = ("综合能力", "MMLU-Pro, GPQA Diamond, C-Eval, CMMLU, AGIEval")
    D2 = ("编程", "HumanEval, MBPP, LiveCodeBench, SWE-Bench, BigCodeBench")
    D3 = ("数学推理", "MATH, GSM8K, AIME, IMO 2025")
    D4 = ("科学推理", "GPQA Diamond, ARC-Challenge")
    D5 = ("长文本", "LongBench, Needle-in-Haystack")
    D6 = ("指令遵循", "IFEval, AlpacaEval 2.0")
    D7 = ("工具调用/Agent", "BFCL, ToolBench, AgentBench, BrowseComp")
    D8 = ("中文理解", "C-Eval, CMMLU, SuperCLUE")
    D9 = ("多轮对话", "MT-Bench, Arena Elo")

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def benchmarks(self) -> str:
        return self.value[1]


class Profile(Enum):
    """Agent 角色画像."""
    CODER = "coder"
    PLANNER = "planner"
    WRITER = "writer"
    RESEARCHER = "researcher"
    REVIEWER = "reviewer"
    DOUBTER = "doubter"

    @property
    def label(self) -> str:
        labels = {
            "coder": "Coder — 编程开发",
            "planner": "Planner — 任务规划",
            "writer": "Writer — 文档撰写",
            "researcher": "Researcher — 信息调研",
            "reviewer": "Reviewer — 质量审查",
            "doubter": "Doubter — 质疑审视",
        }
        return labels[self.value]

    @property
    def description(self) -> str:
        descs = {
            "coder": "代码生成、调试、测试编写。高权重维度: 编程(30%), 工具调用(20%)",
            "planner": "任务分解、依赖分析、架构规划。高权重维度: 综合能力(25%), 指令遵循(20%)",
            "writer": "文档撰写、翻译、内容创作。高权重维度: 中文理解(25%), 综合能力(20%)",
            "researcher": "信息搜索、数据收集、调研分析。高权重维度: 工具调用(25%), 综合能力(20%)",
            "reviewer": "代码审查、质量把关、输出验证。高权重维度: 编程(25%), 综合能力(20%)",
            "doubter": "质疑假设、发现漏洞、压力测试。高权重维度: 数学推理(20%), 科学推理(20%)",
        }
        return descs[self.value]


@dataclass
class ModelInfo:
    """模型基本信息."""
    name: str
    provider: str
    parameters: Optional[str] = None  # e.g. "7450B (44B active, MoE)"
    context_length: Optional[str] = None  # e.g. "203K"
    is_open_source: bool = False
    release_date: Optional[str] = None
    notes: str = ""


@dataclass
class ModelScores:
    """模型在各维度的得分."""
    model: ModelInfo
    scores: dict[Dimension, Optional[float]] = field(default_factory=dict)
    sources: dict[Dimension, str] = field(default_factory=dict)  # 数据来源

    @property
    def coverage(self) -> float:
        """数据覆盖率 (0.0-1.0)."""
        total = len(Dimension)
        available = sum(1 for v in self.scores.values() if v is not None)
        return available / total

    @property
    def available_dimensions(self) -> list[Dimension]:
        return [d for d, v in self.scores.items() if v is not None]

    @property
    def missing_dimensions(self) -> list[Dimension]:
        return [d for d, v in self.scores.items() if v is None]


@dataclass
class ProfileWeights:
    """某 Profile 在 9 个维度的权重分配."""
    profile: Profile
    weights: dict[Dimension, float] = field(default_factory=dict)

    def __post_init__(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"权重合计应为 1.0，当前为 {total}")

    def normalized_weights(self, available_dims: list[Dimension]) -> dict[Dimension, float]:
        """归一化：只对可用维度重新分配权重."""
        available_weight = sum(self.weights[d] for d in available_dims)
        if available_weight == 0:
            return {}
        return {d: self.weights[d] / available_weight for d in available_dims}


@dataclass
class ScoredModel:
    """打分后的模型结果."""
    model: ModelInfo
    profile: Profile
    dimension_scores: dict[Dimension, Optional[float]]
    weighted_score: float
    coverage: float
    data_quality: str  # e.g. "⭐⭐⭐ 完整", "⚠️ 67%"

    @property
    def summary(self) -> str:
        dims_str = ", ".join(
            f"{d.name}={s:.1f}"
            for d, s in sorted(self.dimension_scores.items(), key=lambda x: (x[1] is None, -(x[1] or 0)))
            if s is not None
        )
        return f"{self.model.name}: {self.weighted_score:.1f} ({self.coverage:.0%}) — {dims_str}"
