# LLM Eval Workbench 🧪

> **大模型能力评估工作台** - 多维度 benchmark 评分系统 + 角色画像权重矩阵 + **可学习权重**（偏好数据驱动）+ 本地模型部署监控 + MCP Server 工具链

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![uv](https://img.shields.io/badge/uv-managed-purple)](https://docs.astral.sh/uv/)
[![Tests](https://img.shields.io/badge/tests-49%2F49-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📊 核心能力

```
                    ┌──────────────────────────┐
                    │    LLM Eval Workbench     │
                    └──────────┬───────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  评测引擎     │   │  角色画像     │   │  部署监控     │
   │  scoring.py  │   │  profiles.py │   │  local-deploy │
   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
          │                  │                  │
          ▼                  ▼                  ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │ 9 维度评分    │   │ 6 角色权重    │   │ MLX/omlx     │
   │ 缺失数据处理  │   │ 归一化计算    │   │ 健康检查 API  │
   │ 排名+报告     │   │ 交叉矩阵      │   │ 延迟监控     │
   └──────────────┘   └──────────────┘   └──────────────┘
```

### 九大能力维度

| 维度 | 名称 | 代表评测集 |
|------|------|-----------|
| D1 | 综合能力 | MMLU-Pro, GPQA Diamond, C-Eval |
| D2 | 编程 | SWE-Bench, LiveCodeBench, HumanEval |
| D3 | 数学推理 | AIME, MATH, GSM8K, IMO 2025 |
| D4 | 科学推理 | GPQA Diamond, ARC-Challenge |
| D5 | 长文本 | LongBench, Needle-in-Haystack |
| D6 | 指令遵循 | IFEval, AlpacaEval 2.0 |
| D7 | 工具调用/Agent | BFCL, AgentBench, BrowseComp |
| D8 | 中文理解 | C-Eval, CMMLU, SuperCLUE |
| D9 | 多轮对话 | MT-Bench, Arena Elo |

### 六个 Agent 角色画像

| Profile | 高权重维度 | 典型场景 |
|---------|-----------|---------|
| **Coder** | 编程 30%, 工具调用 20% | 代码生成、调试、测试 |
| **Planner** | 综合 25%, 指令遵循 20% | 任务分解、依赖分析 |
| **Writer** | 中文 25%, 综合 20% | 文档撰写、内容创作 |
| **Researcher** | 工具调用 25%, 综合 20% | 信息搜索、数据收集 |
| **Reviewer** | 编程 25%, 综合 20% | 代码审查、质量把关 |
| **Doubter** | 数学 20%, 科学 20% | 质疑假设、压力测试 |

---

## 🎓 可学习权重（新）

手工权重矩阵有内在逻辑，但**未经真实偏好数据校验**。本模块用公开 pairwise 偏好数据学习每个 Profile 的维度权重，输出带 95% 置信区间的可解释权重，**显著超越手工才上线**。

### 技术路径（对应面试题修正后的方案）

| 设计决策 | 选择 | 理由 |
|---|---|---|
| 潜变量处理 | **因子分析**（非 transformer） | 9 维有共线（D1↔D8），显式降维比隐式拟合更匹配小数据 |
| 模型 | **线性 Bradley-Terry** | 模型数才几十个，transformer 容量被方差吞掉 |
| 推断 | **NumPyro 贝叶斯**（NUTS 采样） | 得真后验，天然带置信区间 |
| 先验 | **Logistic-Normal**，手工权重作锚 | 数据不足时退化为手工权重（安全） |
| 验收 | **held-out lift ≥ 5pp** 才 USE_LEARNED | 不达标诚实 fallback 到手工 |

### 已验证（合成数据）

```
sanity check: MAE 0.0388, 相关性 0.895, 收敛 ✓
全 Profile:   5/6 USE_LEARNED (Planner 正确 FALLBACK, lift 3.5pp < 5pp)
测试:         49/49 通过
```

### 真实数据状态

详见 [`docs/real-data-feasibility.md`](docs/real-data-feasibility.md)。简述：
- ✅ 公开 Elo leaderboard 已下载（383 模型，`data/arena_text_elo_clean.csv`）
- ✅ 4 个真实模型 benchmark 分数已存档（`data/model_scores_expanded.csv`）
- 🔒 **Arena pairwise 对话数据是 gated repo**，需申请 HF token 才能学权重（唯一可行路径）
- ❌ Elo 回归验证不可行（benchmark 代际断崖，仅 3 模型能对齐，需 15+）

---

## 🚀 快速开始

```bash
# 安装 (用 uv 管理环境; 见 docs/environment-setup.md)
uv sync --group learning --group dev

# 运行内置 demo（4 个模型 × 6 个 Profile）
uv run python scripts/evaluate.py

# 自定义数据
uv run python scripts/evaluate.py --data data/model_scores.csv --output my_report.md

# 只看特定 Profile
uv run python scripts/evaluate.py --profile coder,writer --format compact

# 用学出权重 (需先跑 learn_weights.py)
uv run python scripts/evaluate.py --weights auto
```

> 💡 权重学习子模块（`src/llm_eval/learning/`）需要 `--group learning` 依赖。
> 不装时核心评分功能（默认手工权重）仍可用：`uv sync && uv run python scripts/evaluate.py`

### 编程 API

```python
from llm_eval.models import Dimension, ModelInfo, ModelScores, Profile
from llm_eval.profiles import get_profile_weights
from llm_eval.scoring import compute_weighted_score

model = ModelScores(
    model=ModelInfo(name="My-Model", provider="custom"),
    scores={Dimension.D1: 85.0, Dimension.D2: 78.0},
)
result = compute_weighted_score(model, get_profile_weights(Profile.CODER))
print(f"Coder 得分: {result.scored.weighted_score:.1f}")
```

更多示例见 [`examples/basic_usage.py`](examples/basic_usage.py)。

---

## 📈 核心结论

> ⚠️ 以下为演示数据（BenchLM 聚合分）下的结果，非真实评测结论。
> 当前代码复算值（`uv run python scripts/evaluate.py` 可复现）：

| # | 模型 | 综合排名 | Coder | Planner | Researcher |
|---|------|---------|-------|---------|------------|
| 🥇 | **GLM-5.1** | 最全面 | 84.1 | 84.7 | 80.7 |
| 🥈 | **DeepSeek-V4-Pro** | 编程/数学最强 | 84.5 | 84.5 | 78.4 |
| 🥉 | DeepSeek-V4-Flash | 高性价比 | 75.0 | 76.3 | 69.5 |

> **方法论警示**：DS-V4-Pro 缺 D6/D9 维度，当前评分采用"跳过缺失 + 权重归一化"，
> 这会让缺数据的模型在 Coder/Reviewer/Doubter 场景系统性虚高。详见本页下方"缺失数据处理策略"。

---

## 📂 项目结构

```
llm-eval-workbench/
├── src/llm_eval/           # 核心评测引擎
│   ├── models.py           #   数据模型 (Dimension/Profile/ModelScores)
│   ├── benchmarks.py       #   Benchmark 定义与权重
│   ├── profiles.py         #   6 个 Profile 的手工权重矩阵
│   ├── scoring.py          #   加权评分引擎
│   ├── report.py           #   Markdown 报告生成 (支持学出权重 CI 标注)
│   └── learning/           # 🎓 可学习权重子包 (需 --group learning)
│       ├── data.py         #   Arena 下载 + 合成数据 + 统一 schema
│       ├── profiling.py    #   LLM 软标注 + Cohen's κ 质量校验
│       ├── features.py     #   共线性诊断 + 因子分析 + 交叉项
│       ├── model.py        #   NumPyro Bradley-Terry 贝叶斯模型
│       ├── train.py        #   训练编排 + sanity check
│       ├── evaluate.py     #   held-out 评估 + 5pp 验收门槛
│       └── persist.py      #   权重序列化 + drop-in 接口
├── scripts/
│   ├── evaluate.py         # CLI: 评分 + 报告 (--weights auto/learned/handcraft)
│   ├── learn_weights.py    # CLI: 权重学习 (--source synthetic/arena)
│   └── benchmark/          # 推理基准测试（MLX vs GGUF 等）
├── examples/
│   └── basic_usage.py      # 编程 API 使用示例
├── tests/                  # pytest 测试 (49 个, 含 ML 跳过标记)
├── docs/                   # 评测方法论与报告
│   ├── benchmarks-survey.md
│   ├── benchmark-results.md
│   ├── profile-weight-system.md
│   ├── final-report.md
│   ├── ml-framework-comparison.md
│   ├── model-evaluation-guide.md
│   ├── environment-setup.md      # uv 环境配置 + 网络排查
│   └── real-data-feasibility.md  # 真实数据可行性报告
├── local-deploy/           # MLX 本地部署监控
├── mcp-servers/            # MCP Server 工具链
├── data/                   # 示例数据 + 真实数据存档
│   ├── model_scores.csv             # 原始 4 模型 (旧)
│   ├── arena_text_elo_clean.csv     # 383 模型真实 Elo (Arena 公开)
│   └── model_scores_expanded.csv    # 4 模型真实 benchmark (官方来源)
├── tools/benchmark-weights.yaml
├── pyproject.toml          # PEP 735 dependency-groups (uv 原生)
├── uv.toml                 # uv 镜像配置 (清华源)
├── uv.lock                 # 锁定依赖 (62 包)
└── .python-version         # 固定 3.12
```

---

## 🔬 方法论要点

1. **加权评分**：每个 Profile 定义 9 维度权重矩阵，Σ = 1.0
2. **缺失数据处理**：跳过缺失维度，权重归一化到可用维度
3. **覆盖率标注**：每个模型标注数据完整性
4. **局限性透明**：报告明确列出方法论局限

---

## 🖥️ 本地部署监控 (local-deploy)

跨网络监控 Apple Silicon 上的 MLX 推理服务：

```
ZeroTier 虚拟网络 (192.168.192.0/24)
        │
        ▼
┌───────────────────┐     FastAPI (port 8080)     ┌──────────────────┐
│  Mac M1 Ultra     │ ◄──────────────────────────► │  Linux 监控节点   │
│  oMLX :8000       │     /api/omlx/health        │  llm-eval        │
│  Qwen3-E 0.6B/8B  │     /api/omlx/models        │  workbench       │
│  Qwen3.5-9B       │     /api/omlx/latency       │                  │
└───────────────────┘                             └──────────────────┘
```

---

## 🔧 硬件环境

- **Mac M1 Ultra** (128GB Unified Memory)
- **macOS 26.5.2**
- Python 3.12 + PyTorch (MPS)
- MLX v0.31.x

## 📄 License

MIT © Zjrua
