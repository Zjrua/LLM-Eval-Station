# LLM Eval Workbench 🧪

> **大模型能力评估工作台** — 多维度 benchmark 评分系统 + 角色画像权重矩阵 + 本地模型部署监控

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
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

## 🚀 快速开始

```bash
# 安装
pip install -e .

# 运行内置 demo（4 个模型 × 6 个 Profile）
python scripts/evaluate.py

# 自定义数据
python scripts/evaluate.py --data data/model_scores.csv --output my_report.md

# 只看特定 Profile
python scripts/evaluate.py --profile coder,writer --format compact
```

### 编程 API

```python
from llm_eval.models import Dimension, ModelInfo, ModelScores, Profile
from llm_eval.profiles import get_profile_weights
from llm_eval.scoring import compute_weighted_score

model = ModelScores(
    model=ModelInfo(name="My-Model", provider="custom"),
    scores={
        Dimension.D1: 85.0, Dimension.D2: 78.0,  # ...
    },
)
result = compute_weighted_score(model, get_profile_weights(Profile.CODER))
print(f"Coder 得分: {result.scored.weighted_score:.1f}")
```

更多示例见 [`examples/basic_usage.py`](examples/basic_usage.py)。

---

## 📈 核心结论（基于公开评测数据）

| # | 模型 | 综合排名 | Coder | Planner | Researcher |
|---|------|---------|-------|---------|------------|
| 🥇 | **GLM-5.1** | 最全面 | 81.6 | 82.3 | 80.4 |
| 🥈 | **DeepSeek-V4-Pro** | 编程/数学最强 | 84.5 | 84.7 | 80.8 |
| 🥉 | DeepSeek-V4-Flash | 高性价比 | 74.2 | 77.1 | 69.3 |

> ⚠️ **方法论局限**：缺失数据维度采用「跳过归一化」策略，可能高估或低估模型。详细分析见 [`docs/final-report.md`](docs/final-report.md)。

---

## 📂 项目结构

```
llm-eval-workbench/
├── src/llm_eval/           # 核心评测引擎
│   ├── models.py           #   数据模型（Model, Dimension, Profile）
│   ├── benchmarks.py       #   Benchmark 定义与权重
│   ├── profiles.py         #   6 个 Profile 的权重矩阵
│   ├── scoring.py          #   加权评分引擎
│   └── report.py           #   Markdown 报告生成器
├── scripts/
│   └── evaluate.py         # CLI 命令行入口
├── examples/
│   └── basic_usage.py      # 编程 API 使用示例
├── docs/                   # 📖 评测方法论与报告
│   ├── methodology.md      #   （见下方说明）
│   ├── benchmarks-survey.md
│   ├── benchmark-results.md
│   ├── profile-weight-system.md
│   └── final-report.md
├── local-deploy/           # 🖥️ MLX 本地部署监控
│   ├── omlx_client.py      #   OMLX 异步客户端
│   ├── omlx_monitor.py     #   FastAPI 监控路由
│   ├── app.py              #   独立部署入口
│   └── test_omlx.py        #   pytest 测试
├── data/                   # 📊 示例数据
│   ├── model_scores.csv
│   └── profile_weights.csv
└── pyproject.toml
```

> `docs/methodology.md` 由 `scripts/evaluate.py` 运行时自动生成，或调用 `report.py` 的 `generate_report()` API 生成。

---

## 🔬 方法论要点

1. **加权评分**：每个 Profile 定义 9 维度权重矩阵，Σ = 1.0
2. **缺失数据处理**：跳过缺失维度，权重归一化到可用维度（避免"缺数据=能力差"偏差）
3. **覆盖率标注**：每个模型标注数据完整性（⭐⭐⭐ 完整 / ❌ 数据不足）
4. **局限性透明**：报告明确列出方法论局限（维度重叠、缺失偏差等）

详细方法论见 [`docs/benchmarks-survey.md`](docs/benchmarks-survey.md)。

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
│  Qwen3.5-9B        │     /api/omlx/latency       │                  │
└───────────────────┘                             └──────────────────┘
```

---

## 📄 License

MIT © Zjrua
