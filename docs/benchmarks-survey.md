# 大模型评测集体系与能力维度

> 调研时间：2026-05-26

---

## 一、综合能力

| 评测集 | 版本 | 能力维度 | 题目数量 | 题目格式 | CoT/Non-CoT | 典型评测设置 |
|--------|------|----------|----------|----------|-------------|-------------|
| **MMLU** | 2020（MMLU-Redux 2024） | 综合能力 | 14,042（57 学科） | 4 选 1 多选题 | Non-CoT（5-shot）为主 | 5-shot, 偶见 0-shot |
| **MMLU-Pro** | v1.1 (2024) | 综合能力 | ~14,108（57 学科） | 10 选 1 多选题 | CoT 常用（难度更高） | 5-shot; 也报 0-shot/CoT |
| **C-Eval** | 2023 | 综合能力、中文理解 | 13,948（52 学科） | 4 选 1 多选题（中文） | Non-CoT（5-shot）为主 | 5-shot |
| **CMMLU** | 2023 | 综合能力、中文理解 | 11,541（67 学科） | 4 选 1 多选题（中文） | Non-CoT（5-shot）为主 | 5-shot / 0-shot |
| **AGIEval** | 2023 | 综合能力、推理 | 8,062（20 项任务） | 多选题为主（4 选 1），部分开放式 | 0-shot CoT 常用 | 0-shot CoT（模拟考试场景） |

---

## 二、编程能力

| 评测集 | 版本 | 能力维度 | 题目数量 | 题目格式 | CoT/Non-CoT | 评测指标 |
|--------|------|----------|----------|----------|-------------|---------|
| **HumanEval** | 2021；HumanEval+ 2023 | 编程 | 164 题 | 函数级代码生成（Python） | Non-CoT | pass@1 / pass@10 |
| **MBPP** | 2021；MBPP+ 2023 | 编程 | 974 题（标准集 ~500） | 函数级代码生成（Python） | Non-CoT | pass@1 |
| **LiveCodeBench** | v5 (2025) | 编程 | 700+ 题（持续增长） | 竞赛编程题目（Codeforces/AtCoder 新题） | Non-CoT | pass@k / 严格准确率 |
| **BigCodeBench** | v1.1 (2024) | 编程 | ~600 题（Hard 子集 308） | 函数级代码生成（含库调用） | Non-CoT | pass@k |
| **Codeforces Rating** | 持续更新 | 编程 | 不固定（每次竞赛 4-6 题） | 完整竞赛题解 | Non-CoT | 估算评级（Elo-like） |

---

## 三、数学推理

| 评测集 | 版本 | 能力维度 | 题目数量 | 题目格式 | CoT/Non-CoT | 备注 |
|--------|------|----------|----------|----------|-------------|------|
| **MATH** | 2021 (Hendrycks) | 数学推理 | 12,500（训练 7,500 + 测试 5,000） | 自由作答（7 学科，5 级难度） | **CoT 为标准设置** | Non-CoT 表现显著更低 |
| **GSM8K** | 2021 (OpenAI) | 数学推理 | 8,500（训练 7,500 + 测试 1,000） | 自由作答（小学数学应用题） | **CoT 为标准设置** | 2-8 步算术推理 |
| **AIME 2024/2025** | 年度更新 | 数学推理 | 每年 30 题（I + II 各 15） | 自由作答（整数答案 0-999） | **CoT 必需** | 前沿推理基准，难度极高 |

---

## 四、科学推理

| 评测集 | 版本 | 能力维度 | 题目数量 | 题目格式 | CoT/Non-CoT | 备注 |
|--------|------|----------|----------|----------|-------------|------|
| **GPQA Diamond** | 2023 | 科学推理 | 448 题（Diamond 子集） | 4 选 1 多选题（博士级） | CoT/Non-CoT 均可 | "Google-Proof"，答案不易搜索 |
| **ARC-Challenge** | 2018 (AI2) | 科学推理 | 2,597 题（Challenge 集） | 4 选 1 多选题 | CoT/Non-CoT 均可 | 科学常识推理 |

---

## 五、长文本

| 评测集 | 版本 | 能力维度 | 题目数量 | 题目格式 | CoT/Non-CoT | 备注 |
|--------|------|----------|----------|----------|-------------|------|
| **LongBench** | 2023 v1 | 长文本 | 4,750（21 数据集，6 任务类别） | 多选 + 开放式生成 | Non-CoT 为主 | 最长 32k tokens；含中英文 |
| **Needle-in-Haystack** | 2024 (Greg Kamradt) | 长文本 | 30-100+（可扩展） | 事实检索（从长文档中找特定信息） | Non-CoT | 二值正确率；按深度和长度报告 |

---

## 六、指令遵循

| 评测集 | 版本 | 能力维度 | 题目数量 | 题目格式 | CoT/Non-CoT | 备注 |
|--------|------|----------|----------|----------|-------------|------|
| **IFEval** | 2024 | 指令遵循 | 541 题 | 自由文本（含可验证约束） | Non-CoT | 严格/宽松准确率 |
| **AlpacaEval 2.0** | 2023 (更新) | 指令遵循 | 805 题 | 单轮开放式生成 | Non-CoT | GPT-4 配对比较（Win Rate） |

---

## 七、工具调用 / Agent 能力

| 评测集 | 版本 | 能力维度 | 题目数量 | 题目格式 | CoT/Non-CoT | 备注 |
|--------|------|----------|----------|----------|-------------|------|
| **BFCL** | 2024 (持续更新) | 工具调用 | 1,500+（多类别） | NL 请求 → 函数调用 JSON | Non-CoT | 函数名+参数精确匹配 |
| **ToolBench** | 2024 v1 | 工具调用、Agent | 1,000+ 评估任务（16,000 训练） | 多步工具调用序列 | Non-CoT | Pass Rate + GPT-4 Win Rate |

---

## 八、中文能力

| 评测集 | 版本 | 能力维度 | 题目数量 | 题目格式 | CoT/Non-CoT | 备注 |
|--------|------|----------|----------|----------|-------------|------|
| **C-Eval** | 2023 | 中文理解、综合能力 | 13,948 | 4 选 1 多选题（中文） | Non-CoT（5-shot） | 见"综合能力"部分 |
| **CMMLU** | 2023 | 中文理解、综合能力 | 11,541 | 4 选 1 多选题（中文） | Non-CoT（5-shot） | 含中国文化特有学科 |
| **SuperCLUE** | 2023/2024 | 中文理解 | 1,000+（v1） | 多选 / 判断 / 简答（中文） | Non-CoT | 覆盖推理、知识、安全 |

---

## 九、多轮对话

| 评测集 | 版本 | 能力维度 | 题目数量 | 题目格式 | CoT/Non-CoT | 备注 |
|--------|------|----------|----------|----------|-------------|------|
| **MT-Bench** | 2023 | 多轮对话 | 80 轮对话（2 轮/对话，共 160 题） | 8 类别多轮对话 | Non-CoT | GPT-4 评委（配对+单分） |
| **AlpacaEval 2.0** | 2023 | 多轮对话 | 805 题 | 单轮指令遵循 | Non-CoT | 亦用于对话质量评估 |

---

## 总结：能力维度 × 评测集对照

| 能力维度 | 评测集 |
|----------|--------|
| 综合能力 | MMLU, MMLU-Pro, C-Eval, CMMLU, AGIEval |
| 编程 | HumanEval, MBPP, LiveCodeBench, BigCodeBench, Codeforces |
| 数学推理 | MATH, GSM8K, AIME |
| 科学推理 | GPQA Diamond, ARC-Challenge |
| 长文本 | LongBench, Needle-in-Haystack |
| 指令遵循 | IFEval, AlpacaEval 2.0 |
| 工具调用/Agent | BFCL, ToolBench |
| 中文理解 | C-Eval, CMMLU, SuperCLUE |
| 多轮对话 | MT-Bench, AlpacaEval 2.0 |

---

## 搜索来源

- MMLU: arXiv 2009.03300; MMLU-Pro: arXiv 2406.01574
- C-Eval: arXiv 2305.08322; CMMLU: arXiv 2306.09212
- AGIEval: arXiv 2304.06364
- HumanEval: arXiv 2107.03374; MBPP: arXiv 2108.07732
- LiveCodeBench: arXiv 2403.07875; BigCodeBench: arXiv 2407.14619
- MATH: arXiv 2103.03874; GSM8K: arXiv 2110.14168
- AIME: American Invitational Mathematics Examination official
- GPQA Diamond: arXiv 2311.12022
- ARC-Challenge: arXiv 1803.05457
- LongBench: arXiv 2308.14508
- Needle-in-Haystack: Greg Kamradt blog (2024)
- IFEval: Google Research GitHub; arXiv 2311.07911
- AlpacaEval 2.0: tatsu-lab/alpaca_eval GitHub
- BFCL: ShishirPatil/gorilla GitHub
- ToolBench: arXiv 2307.16789; OpenBMB/ToolBench GitHub
- SuperCLUE: CLUEbenchmark/SuperCLUE GitHub
- MT-Bench: lm-sys/FastChat; arXiv 2306.05685
