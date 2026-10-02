# Station Harness v2 — 执行验证型 Agent 评测（DeepSWE 范式）

> 设计参考：DeepSWE（datacurve-ai）、SWE-Bench Verified、mini-swe-agent 的评测哲学。
> 核心转变：v1 是「聚合别人跑的分数」；v2 是「自己设计任务、自己写 verifier、自己跑 agent、自己出统计」——一级实证。

## 设计原则（从 DeepSWE 借鉴）

1. **任务三件套**：每个任务 = `instruction.md`（agent 读）+ `verify.sh`（可执行 verifier）+ `solution/`（参考解，仅人工审核用，打分时永不使用）
2. **Prompt-verifier 双射**：verifier 检查的必须恰好是 instruction 要求的——测多假阴性，测少假阳性
3. **接受广度**：verifier 断言可观测行为（CLI 输出、文件内容、退出码），不断言内部实现
4. **回归检查**：verify.sh 同时跑「必须通过的检查」（fail→pass）和「不能破坏的检查」（pass→pass）
5. **Verifier 防抖**：任务入库前 verifier 对参考解跑 3 遍，结果不一致打回
6. **Harness 固定**：所有被测模型走同一个 agent loop、同一套工具（read/write/edit/bash）、同一个系统提示——分数反映模型能力，不反映脚手架
7. **统计口径**：`--repeats k` 多次采样，pass@k（无偏估计）+ pass^k（可靠性）+ Wilson 95% CI；复用 LearnLoop eval/stats.py 已验证的口径

## 任务格式

```
tasks/<task_id>/
├── instruction.md     # 给 agent 的任务描述（中文，含验证方式提示）
├── setup/             # 任务初始环境（由 harness 复制到临时工作目录）
│   ├── files...       # 待修改的代码/数据
│   └── ... 
├── verify.sh          # verifier：判定+回归检查。exit 0 = pass
└── solution/          # 参考解（验证 verifier 可满足；不入工作目录）
```

## v1 的分场景选型系统处置

**整体舍弃**：9 维度×6 角色权重矩阵、52 benchmark 聚合、可学习权重（Bradley-Terry）全部移入 `legacy/` 归档，不再维护。理由：二级聚合无法回答「这个模型在我的任务上到底行不行」，且权重校准依赖的 Arena 偏好数据与真实 Agent 场景分布不同。

保留并迁移：本地部署监控（MLX/oMLX 对比）、MCP 封装（Unlimited-OCR 7 工具）、统计工具（Wilson CI / pass@k）。

## Pilot 规模

20 个任务 × 3 模型（deepseek-flash / glm-5.3-flash / kimi-for-coding，均走本机 new-api 网关）× 3 次采样。任务来源：LearnLoop-agent 真实代码库的 bug-fix / feature / refactor 三类。
