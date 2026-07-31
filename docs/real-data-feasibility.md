# 真实偏好数据可行性报告

> 生成时间：2026-07-31
> 目的：记录用真实 LMSYS Chatbot Arena 数据学习权重的可行性验证结果

## 一、验证过的事实

### 1.1 Arena 数据集访问

| 数据集 | 访问性 | 用途 |
|---|---|---|
| `lmsys/chatbot_arena_conversations` | 🔒 **gated**（需 HF 账号 + 申请权限 + token） | 33 万场 pairwise 对话全文，是学维度权重的**必需**数据 |
| `lmarena-ai/leaderboard-dataset` | ✅ **公开**（已下载） | 已聚合的 Elo rating + 置信区间，46 文件/22 类别 |

**关键区分**：
- pairwise 对话数据（gated）才能学"维度分差 → 胜负"的权重
- Elo leaderboard（公开）只能验证"哪些维度与总强度相关"，不能学权重

### 1.2 Elo leaderboard 实测

从 `lmarena-ai/leaderboard-dataset` 下载 text 类别，去重后得到 **383 个可信模型**（vote ≥ 500）的 Elo 评分（2026-07-30 快照）。已存档 `data/arena_text_elo_clean.csv`。

Top 模型（Elo）：claude-opus-5-max 1512、claude-opus-4-6-thinking 1503、gemini-3.1-pro 1479、glm-5.1 1465、deepseek-v4-pro 1450。

### 1.3 模型分数对齐（核心障碍）

为做 Elo ~ 维度分回归，需"模型名 + Elo + 9 维度分"三者对齐。实测查到 4 个真实模型的官方 benchmark 分数：

| 模型 | Elo | 维度对齐 | harness |
|---|---|---|---|
| deepseek-v3 | 1333 | 6/9 | 旧代（MMLU-Pro/HumanEval/MATH） |
| llama-3.1-405b | 1284 | 5/9 | 旧代 |
| llama-3.1-70b | 1261 | 5/9 | 旧代 |
| glm-5.1 | 1465 | 1/9 | **新代**（SWE-Bench-Pro/τ³-Bench/AIME2026） |

## 二、为什么 Elo 回归不可行（诚实结论）

**3 个模型做 9 维回归是数学上无效的**（欠定 + 严重过拟合，R² 必然虚高，结论不可信）。

更根本的障碍是 **benchmark 代际断崖**：
- 2024 旧代模型（Llama-3.1/DeepSeek-V3）报 MMLU/HumanEval/MATH
- 2026 新代模型（GLM-5.1/Claude-Opus-5）报 SWE-Bench-Pro/τ³-Bench/AIME-2026
- 两代交集仅 GPQA，分数天然不可比

**继续检索不会改变这个结构**——这是行业 benchmark 体系的代际更替，非检索努力能解决。混采两代分数做回归，等于犯"评测条件不统一"的错误（正是评测方法论要避免的）。

## 三、可行路径

### 路径 A：pairwise 权重学习（需 HF token，推荐）

这是原始计划，也是**唯一能学维度权重**的真实数据路径：

1. 去 https://huggingface.co/datasets/lmsys/chatbot_arena_conversations 申请访问（通常自动批准）
2. Settings → Access Tokens → 创建 read token
3. 把 token 给我，我带 `HF_TOKEN` 环境变量重下 33 万场对话
4. 用已实现的 pipeline：`load_arena_data → profiling.annotate_samples → train_profile → evaluate`

**为什么这条路可行**：BT 模型从单场"A vs B 胜负"学权重，每场对话自带模型名，只要有维度分的模型对（4 个真实模型能产生 C(4,2)=6 对）就能启动，不像回归需要 15+ 模型对齐。

### 路径 B：合成数据（已验证，立即可用）

`uv run python scripts/learn_weights.py --source synthetic` 已跑通，49/49 测试通过，5/6 profile USE_LEARNED。这是**当前可交付的真实成果**，虽不基于真实偏好数据。

## 四、已存档的真实数据资产

| 文件 | 内容 | 来源 |
|---|---|---|
| `data/arena_text_elo_clean.csv` | 383 模型 Elo（2026-07-30） | lmarena-ai/leaderboard-dataset (公开) |
| `data/model_scores_expanded.csv` | 4 模型真实 benchmark 分数 | 各官方 HF model card |

这两个文件是后续 pairwise 学习（路径 A）的现成输入，拿到 HF token 后可直接用。
