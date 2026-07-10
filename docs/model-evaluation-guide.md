# 模型评测体系：9维度 × 6角色加权评分

## 概述

从模型能力评测到实际部署配置的完整决策框架。

## 9 维度体系

| Dim | 名称 | 代表评测集 | 总权重 |
|-----|------|-----------|--------|
| D1 | 综合能力 | MMLU-Pro, GPQA Diamond, Chinese SimpleQA, SimpleQA, HLE | 10.0 |
| D2 | 编程 | SWE-Bench Verified, LiveCodeBench, Terminal Bench, NL2Repo 等 | 12.0 |
| D3 | 数学推理 | AIME 2026, IMOAnswerBench, HMMT Feb 2026 | 7.0 |
| D4 | 科学推理 | （暂并入 D1，待扩展） | — |
| D5 | 长文本 | LongBench-V2, MRCR 1M, CorpusQA 1M | 5.0 |
| D6 | 指令遵循 | GDPval rubrics, SpreadSheetBench | 3.0 |
| D7 | 工具调用/Agent | BrowseComp, MCP Atlas, Toolathlon, τ²-Bench 等 | 11.0 |
| D8 | 中文理解 | C-Eval, CMMLU（待数据充足） | — |
| D9 | 多轮对话 | MT-Bench, AlpacaEval（待数据充足） | — |

## 6 个角色 Profile

| 维度 | Coder | Planner | Writer | Researcher | Reviewer | Doubter |
|------|-------|---------|--------|------------|----------|---------|
| D1 综合 | 1 | 0.5 | 1 | 2 | 2 | 2 |
| D2 编程 | 2 | 0.5 | 0 | 0 | 0.5 | 0 |
| D3 数学 | 0 | 1 | 0 | 1 | 1 | 1 |
| D5 长文本 | 1 | 0 | 1 | 1 | 1 | 1 |
| D6 指令遵循 | 1.5 | 2 | 0 | 1 | 1 | 1 |
| D7 工具调用 | 0.5 | 1 | 2 | 1 | 0.5 | 1 |

## 计算方法

**维度分数** = Σ(评测集分数 × 评测集权重) / Σ(权重)
（仅对有数据的评测集求和，缺失数据不拉低分数）

**Profile 分数** = Σ(维度分数 × 维度权重) / Σ(权重)

## 数据来源

- DeepSeek: 官方 HuggingFace 技术报告, deepseekv4.wiki
- GLM/智谱: docs.bigmodel.cn, benchlm.ai
- MiniMax: minimaxi.com/models/text/m3
- Kimi: github.com/MoonshotAI/Kimi-K2
- Qwen: 官方 GitHub benchmark
- GPT/Claude/Gemini: minimaxi.com 对比表, 各官方技术报告
- 其他: benchlm.ai, Papers With Code

## 注意事项

- 非 0-100 分范围的值（如 Codeforces Rating）需过滤
- 维度 < 4/6 有数据时排名不可靠
- 模型名称需通过 API `/v1/models` 验证，不可猜测
