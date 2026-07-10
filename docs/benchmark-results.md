# 各模型在主流评测集上的实测成绩

> 搜集日期：2026-05-26
> 说明：数据来自多个公开来源，已标注来源类型（官方/第三方社区/第三方评测平台）

---

## 一、智谱 GLM-5.1

### 基本信息
- 参数规模：约 7450 亿总参数，440 亿激活参数（MoE, 256专家, 激活8个）
- 上下文长度：203K tokens
- 开源权重：是（Open Weight）
- 发布日期：2026-04-07

### 类别级综合得分（来源: BenchLM.ai, 第三方评测平台）

| 类别 | 得分 | 排名 | 说明 |
|------|------|------|------|
| Overall | 82/100 | #18/117 | 综合评分 |
| Agentic | 80.4/100 | - | Terminal-Bench 2.0, BrowseComp, OSWorld-Verified, GAIA, TAU-bench, WebArena |
| Coding | 83.4/100 | #13/117 | SWE-bench Verified, LiveCodeBench, SWE-bench Pro, SWE-Rebench, SciCode |
| Reasoning | 64.4/100 | #32/117 | MuSR, LongBench v2, MRCRv2, ARC-AGI-2 |
| Knowledge | 84.4/100 | #10/117 | GPQA, SuperGPQA, MMLU-Pro, HLE, FrontierScience, SimpleQA |
| Math | 89.7/100 | #15/117 | AIME 2025, BRUMO 2025, MATH-500, FrontierMath |
| Instruction Following | 91.7/100 | #10/117 | - |
| Arena Elo | 1472 | - | Chatbot Arena |

来源: https://benchlm.ai/models/glm-5-1

### 具体评测集成绩

| 评测集 | 分数 | 来源 | 来源类型 | 备注 |
|--------|------|------|----------|------|
| SWE-Bench Pro | **58.4** | 智谱官方文档 | 官方 | 发布时全球最佳，超过 GPT-5.4、Claude Opus 4.6、Gemini 3.1 Pro |
| MMLU-Pro (EM) | **86.0** | deepseekv4.wiki 对比表 | 第三方社区 | 与 DS-V4-Pro Max 同表对比 |
| GPQA Diamond (Pass@1) | **86.2** | deepseekv4.wiki 对比表 | 第三方社区 | 同上 |
| Toolathlon (Pass@1) | **40.7** | deepseekv4.wiki 对比表 | 第三方社区 | 同上 |

来源: https://docs.bigmodel.cn/cn/guide/models/text/glm-5.1 , https://deepseekv4.wiki/

---

## 二、智谱 GLM-5-Turbo

### 基本信息
- 上下文长度：200K tokens
- 模型类型：闭源（Proprietary）
- 发布日期：2026-03-01
- Arena Elo: 1456

### 评测成绩

| 评测集 | 分数 | 来源 | 来源类型 | 备注 |
|--------|------|------|----------|------|
| 综合评分 | 未排名 | BenchLM.ai | 第三方评测平台 | 仅有17个公开评测分数，不足以获得综合排名 |
| Agentic 类别 | 0.0/100 | BenchLM.ai | 第三方评测平台 | 数据不足 |
| Coding 类别 | 0.0/100 | BenchLM.ai | 第三方评测平台 | 数据不足 |
| Reasoning 类别 | 0.0/100 | BenchLM.ai | 第三方评测平台 | 数据不足 |
| Knowledge 类别 | 0.0/100 | BenchLM.ai | 第三方评测平台 | 数据不足 |
| Math 类别 | 0.0/100 | BenchLM.ai | 第三方评测平台 | 数据不足 |

**说明**: GLM-5-Turbo 在 BenchLM.ai 上有 17 个已发布评测分数（共221个跟踪评测集），但不足以获得类别或全局排名。具体分项分数未在公开页面展示。该模型为闭源模型，官方文档中未公开详细评测表格。

来源: https://benchlm.ai/models/glm-5-turbo

---

## 三、DeepSeek-V4-Flash

### 基本信息
- 参数规模：2840 亿总参数，130 亿激活参数（MoE）
- 上下文长度：1M tokens
- 开源：是（MIT 协议）
- 发布日期：2026-04-24
- 目标对标：GPT-4o-mini, Llama 4 Flash

### Base 模型评测成绩（来源: deepseekv4.wiki, 第三方社区）

| 评测集 | 分数 | 对比 (V4-Base) | 来源类型 | 备注 |
|--------|------|----------------|----------|------|
| HumanEval | **69.5** | 76.8 (V4-Base) | 第三方社区 | V4-Flash-Base vs V4-Pro-Base 对比 |
| MMLU | **88.7** | 90.1 (V4-Base) | 第三方社区 | 同上 |
| MATH | **57.4** | 64.5 (V4-Base) | 第三方社区 | 同上 |
| LongBench-V2 | **44.7** | 51.5 (V4-Base) | 第三方社区 | 同上 |

### Instruct 模型评测成绩（来源: 腾讯云开发者社区文章, 第三方社区, 引用官方技术报告）

| 评测集 | 分数 | 对比 (V4-Pro) | 来源类型 | 备注 |
|--------|------|---------------|----------|------|
| MMLU (世界知识) | **82.1** | 89.7 (Pro) | 第三方社区 | 腾讯云文章引用官方技术报告 |
| GSM8K (数学推理) | **85.6** | 92.4 (Pro) | 第三方社区 | 同上 |
| HumanEval (代码生成) | **76.8** | 85.2 (Pro) | 第三方社区 | 同上 |
| MMBench (多模态) | **78.2** | 83.5 (Pro) | 第三方社区 | 同上 |
| AgentBench (智能体) | **68.5** | 76.8 (Pro) | 第三方社区 | 同上 |

来源: https://deepseekv4.wiki/ , https://cloud.tencent.com/developer/article/2663824
官方技术报告: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

---

## 四、DeepSeek-V4-Pro

### 基本信息
- 参数规模：1.6 万亿总参数，490 亿激活参数（MoE）
- 上下文长度：1M tokens
- 开源：是（MIT 协议）
- 发布日期：2026-04-24
- 目标对标：GPT-5.4, Claude Opus 4.6

### Base 模型评测成绩（来源: deepseekv4.wiki, 第三方社区）

| 评测集 | 分数 | 对比 (V3.2-Base) | 对比 (V4-Flash-Base) | 来源类型 |
|--------|------|-------------------|---------------------|----------|
| HumanEval | **76.8** | 62.8 | 69.5 | 第三方社区 |
| MMLU | **90.1** | 87.8 | 88.7 | 第三方社区 |
| MATH | **64.5** | 60.5 | 57.4 | 第三方社区 |
| LongBench-V2 | **51.5** | 40.2 | 44.7 | 第三方社区 |

### Instruct 模型评测成绩（来源: 腾讯云开发者社区文章, 第三方社区, 引用官方技术报告）

| 评测集 | 分数 | GPT-5.4 (参考) | 来源类型 | 备注 |
|--------|------|----------------|----------|------|
| MMLU (世界知识) | **89.7** | ~90.5 | 第三方社区 | 腾讯云文章引用官方技术报告 |
| GSM8K (数学推理) | **92.4** | ~93.0 | 第三方社区 | 同上 |
| HumanEval (代码生成) | **85.2** | ~86.0 | 第三方社区 | 同上 |
| MMBench (多模态) | **83.5** | ~85.0 | 第三方社区 | 同上 |
| AgentBench (智能体) | **76.8** | ~78.0 | 第三方社区 | 同上 |

### V4-Pro Max 专项评测成绩（来源: deepseekv4.wiki, 第三方社区）

| 评测集 | 分数 | Opus 4.6 Max | GPT-5.4 xHigh | Gemini 3.1 Pro High | GLM-5.1 Thinking | 来源类型 |
|--------|------|-------------|---------------|--------------------|--------------------|----------|
| MMLU-Pro (EM) | **87.5** | 89.1 | 87.5 | 91.0 | 86.0 | 第三方社区 |
| GPQA Diamond (Pass@1) | **90.1** | 91.3 | 93.0 | 94.3 | 86.2 | 第三方社区 |
| LiveCodeBench (Pass@1) | **93.5** | 88.8 | - | 91.7 | - | 第三方社区 |
| Codeforces (Rating) | **3206** | - | 3168 | 3052 | - | 第三方社区 |
| MRCR 1M (MMR) | **83.5** | 92.9 | - | 76.3 | - | 第三方社区 |
| Toolathlon (Pass@1) | **51.8** | 47.2 | 54.6 | 48.8 | 40.7 | 第三方社区 |

来源: https://deepseekv4.wiki/ , https://cloud.tencent.com/developer/article/2663824
官方技术报告: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf

---

## 五、跨模型横向对比汇总

### 基础能力对比（Instruct 版本）

| 评测集 | GLM-5.1 | GLM-5-Turbo | DS-V4-Flash | DS-V4-Pro |
|--------|---------|-------------|-------------|-----------|
| MMLU | 数据缺失 | 数据缺失 | 82.1 | 89.7 |
| MMLU-Pro (EM) | 86.0 | 数据缺失 | 数据缺失 | 87.5 |
| GSM8K | 数据缺失 | 数据缺失 | 85.6 | 92.4 |
| MATH | 数据缺失 | 数据缺失 | 数据缺失 | 64.5 (Base) |
| HumanEval | 数据缺失 | 数据缺失 | 76.8 | 85.2 |
| GPQA Diamond | 86.2 | 数据缺失 | 数据缺失 | 90.1 |
| LiveCodeBench | 数据缺失 | 数据缺失 | 数据缺失 | 93.5 |
| SWE-Bench Pro | 58.4 | 数据缺失 | 数据缺失 | 数据缺失 |
| Codeforces | 数据缺失 | 数据缺失 | 数据缺失 | 3206 |
| Toolathlon | 40.7 | 数据缺失 | 数据缺失 | 51.8 |
| MMBench | 数据缺失 | 数据缺失 | 78.2 | 83.5 |
| AgentBench | 数据缺失 | 数据缺失 | 68.5 | 76.8 |

---

## 六、数据来源汇总

| 来源 | URL | 类型 |
|------|-----|------|
| 智谱官方文档 (GLM-5.1) | https://docs.bigmodel.cn/cn/guide/models/text/glm-5.1 | 官方 |
| BenchLM.ai (GLM-5.1) | https://benchlm.ai/models/glm-5-1 | 第三方评测平台 |
| BenchLM.ai (GLM-5-Turbo) | https://benchlm.ai/models/glm-5-turbo | 第三方评测平台 |
| OpenRouter (GLM-5.1) | https://openrouter.ai/z-ai/glm-5.1 | 第三方 API 平台 |
| GLM-5 社区站 | https://glm-5.org/zh/ | 第三方社区 |
| DeepSeek V4 Wiki | https://deepseekv4.wiki/ | 第三方社区 |
| 腾讯云文章 (DeepSeek V4 解析) | https://cloud.tencent.com/developer/article/2663824 | 第三方社区（引用官方技术报告） |
| DeepSeek 官方技术报告 | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf | 官方 |

---

## 七、数据缺失说明

1. **GLM-5.1 缺失**: MMLU、GSM8K、MATH、HumanEval、MBPP、BBH、ARC-Challenge、HellaSwag、TruthfulQA 等传统评测集的具体分数在公开页面中未以表格形式展示。BenchLM.ai 提供了类别级聚合分数，但未公开每个评测集的具体数值。
2. **GLM-5-Turbo 严重缺失**: 作为闭源模型，公开评测数据极少，BenchLM 上17个分数均不足以产生有效排名。
3. **DeepSeek V4 系列**: 腾讯云文章引用的官方技术报告数据可能与实际存在偏差（该文章为第三方解读），建议以 HuggingFace 上的官方 PDF 技术报告为准进行验证。
4. **CoT/非 CoT 区分**: 本次搜集的数据中，部分为 CoT（如 GSM8K 8-shot CoT），部分为非 CoT（如 MMLU 5-shot），具体设置需查阅原始技术报告确认。
