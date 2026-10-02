"""Benchmark 定义与权重配置."""

from .models import Dimension

# 每个维度下的评测集及其权重
# 权重用于归一化计算维度均值，相对比例决定影响力

BENCHMARK_WEIGHTS: dict[Dimension, dict[str, float]] = {
    Dimension.D1: {
        "MMLU-Pro": 3.0,
        "GPQA Diamond": 2.5,
        "Chinese SimpleQA": 2.0,
        "SimpleQA": 1.5,
        "HLE": 1.0,
    },
    Dimension.D2: {
        "SWE-Bench Verified": 2.5,
        "SWE-Bench Pro": 2.0,
        "LiveCodeBench": 2.0,
        "Terminal Bench": 1.5,
        "NL2Repo": 1.0,
        "VIBE-V2": 1.0,
        "PaperBench": 1.0,
        "SVG-Bench": 0.5,
        "KernelBench Hard": 0.5,
    },
    Dimension.D3: {
        "AIME 2026": 2.5,
        "IMOAnswerBench": 2.5,
        "HMMT Feb 2026": 2.0,
    },
    Dimension.D4: {
        "GPQA Diamond": 1.0,
        "ARC-Challenge": 0.5,
    },
    Dimension.D5: {
        "LongBench-V2": 2.0,
        "MRCR 1M": 1.5,
        "CorpusQA 1M": 1.5,
    },
    Dimension.D6: {
        "IFEval": 1.5,
        "AlpacaEval 2.0": 1.0,
        "GDPval rubrics": 2.0,
        "SpreadSheetBench": 1.0,
    },
    Dimension.D7: {
        "BrowseComp": 2.0,
        "MCP Atlas": 2.0,
        "Toolathlon": 2.0,
        "τ²-Bench": 1.5,
        "BFCL": 2.0,
        "AgentBench": 1.5,
        "OSWorld-verified": 1.0,
        "CyberGym": 0.5,
        "BankerToolBench": 0.5,
    },
    Dimension.D8: {
        "C-Eval": 2.5,
        "CMMLU": 2.5,
        "SuperCLUE": 1.5,
    },
    Dimension.D9: {
        "MT-Bench": 1.0,
        "Arena Elo": 1.5,
    },
}

# 维度描述与选型理由
DIMENSION_NOTES: dict[Dimension, str] = {
    Dimension.D1: "经典权威基准，最广泛使用；MMLU-Pro 难度更高区分度更好",
    Dimension.D2: "SWE-Bench 是业界公认的代码能力黄金标准",
    Dimension.D3: "AIME/IMO 代表顶级数学推理，MATH/GSM8K 覆盖基础到中等难度",
    Dimension.D4: "GPQA Diamond 博士级科学问题，答案不易搜索（Google-Proof）",
    Dimension.D5: "LongBench 覆盖 6 类长文本任务，Needle-in-Haystack 测试检索精度",
    Dimension.D6: "IFEval 可验证约束，AlpacaEval 基于 GPT-4 配对比较",
    Dimension.D7: "BFCL 函数调用精确匹配，AgentBench 多步交互评测",
    Dimension.D8: "C-Eval/CMMLU 覆盖 52-67 个中文类别的综合能力",
    Dimension.D9: "Arena Elo 基于匿名对战的核心指标，MT-Bench 多轮对话质量评",
}
