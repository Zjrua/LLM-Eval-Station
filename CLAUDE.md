# LLM Eval Workbench

Apple Silicon (M1 Ultra) 上大模型本地部署、性能评测与工具链集成的实验项目。

## 项目背景

从实际使用中积累的本地部署经验和评测工具：
- MLX / llama.cpp / Ollama 三框架横向选型
- Qwen3.5-27B 控制变量实测数据
- Unlimited-OCR VLM (3.34B) 的 MCP Server 封装
- 多进程并行 OCR 脚本（4-8 worker，已验证 2.7x-4.1x 加速）
- OCR JSON → LaTeX/Markdown 转换器（已用于 484 页教材 OCR）
- 模型能力 9 维度 × 6 角色 加权评分框架
- **可学习权重**：用偏好数据驱动维度权重（因子分析 + 线性 Bradley-Terry + NumPyro 贝叶斯）

## 关键文件

### 核心评分引擎 (src/llm_eval/)
- `models.py` — Dimension(9)/Profile(6)/ModelScores/ProfileWeights 数据模型
- `profiles.py` — 6 个 Profile 的手工权重矩阵（先验 + fallback）
- `scoring.py` — 加权评分引擎（**已知局限**：缺失数据跳过归一化，可能虚高，见下）
- `report.py` — Markdown 报告生成（支持学出权重 CI 标注）

### 可学习权重 (src/llm_eval/learning/, 需 `--group learning`)
- `data.py` — Arena 数据下载 + 合成数据 + 统一 PairwiseSample schema
- `profiling.py` — LLM 软标注 Profile + Cohen's κ 质量校验
- `features.py` — 共线性诊断(VIF) + 因子分析降维 + 交叉项
- `model.py` — NumPyro Bradley-Terry 贝叶斯模型 (NUTS 采样)
- `train.py` — 训练编排 + sanity check
- `evaluate.py` — held-out 评估 + 5pp 验收门槛
- `persist.py` — 权重序列化 + 转 ProfileWeights drop-in 接口

### CLI
- `scripts/evaluate.py` — 评分 + 报告 (`--weights auto/learned/handcraft`)
- `scripts/learn_weights.py` — 权重学习 (`--source synthetic/arena`)

### 文档
- `docs/environment-setup.md` — uv 环境配置 + 国内网络排查（清华镜像装 Python）
- `docs/real-data-feasibility.md` — 真实偏好数据可行性报告（含 benchmark 代际断崖结论）
- `docs/ml-framework-comparison.md` — MLX vs llama.cpp vs Ollama 对比
- `docs/qwen35-mlx-benchmark.md` — Qwen3.5-27B 实测数据
- `docs/model-evaluation-guide.md` — 评测体系完整文档

### 其他模块
- `mcp-servers/unlimited-ocr-mcp-server.py` — Unlimited-OCR MCP Server（主力）
- `scripts/ocr/parallel_ocr.py` — 并行 OCR
- `scripts/ocr/ocr_to_latex.py` — OCR → LaTeX 转换
- `local-deploy/` — MLX/oMLX 本地部署监控 (FastAPI + ZeroTier)
- `tools/benchmark-weights.yaml` — 评测集权重配置

## 环境

用 uv 管理（见 `docs/environment-setup.md`）：
- Python 固定 3.12 (`.python-version`)，依赖锁在 `uv.lock`
- `uv sync --group learning --group dev` 装依赖
- `uv run pytest tests/` 跑测试（49 个）
- `uv run python scripts/learn_weights.py --source synthetic` 跑权重学习 sanity check
- `uv run python scripts/evaluate.py --weights auto` 用学出权重生成报告
- PyPI 镜像已配清华源 (`uv.toml`)；Python 解释器下载走 GitHub，国内需按文档方案 C 处理

## 已知局限（诚实记录）

1. **缺失数据处理**：`scoring.py` 的"跳过缺失 + 权重归一化"会让缺数据模型虚高。
   DS-V4-Pro 缺 D6/D9，在 Coder/Reviewer/Doubter 上"胜出"在记 0 分时全部翻转。
   这是已知 bug，待修（计划：支持 treat_as_zero / impute 模式）。
2. **权重学习只验证了合成数据**：真实 Arena pairwise 数据是 gated repo，
   需申请 HF token 才能跑真实学习。Elo leaderboard 只能做相关性验证（当前数据不够）。
3. **benchmark 代际断崖**：2024 旧代模型(MMLU/HumanEval)与 2026 新代(SWE-Bench-Pro/τ³-Bench)
   分数不可比，混采等于评测条件不统一。

## 注意

- OCR 模型路径: `~/.cache/modelscope-cache/PaddlePaddle/Unlimited-OCR`（6.21GB）
- MPS patch 是因为上游 modeling_unlimitedocr.py 硬编码 `.cuda()`
- 评测分数数据需从各模型官方技术报告手动收集，脚本只做加权计算
