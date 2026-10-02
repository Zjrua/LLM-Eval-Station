# Station Harness v2 — Pilot 实测结果

> 2026-10-02 · 4 任务 × 3 模型 × 3 采样 · 全部走本机 new-api 网关（temperature=1.0）
> 复现：`python harness/run_eval.py --models deepseek-flash,glm-5.3-flash,kimi-for-coding --repeats 3`

## 任务集（全部通过 verifier 防抖验证：参考解×3 全过 + 空环境必败）

| 任务 | 类型 | 来源 | verifier 组成 |
|------|------|------|---------------|
| t01_fix_intent_classifier | bug-fix | LearnLoop 意图分类器真实 bug 场景 | 隐藏 pytest（6 用例，produce 类 fail→pass）+ 防作弊回归（测试文件未删改） |
| t02_bm25_idf | 实现 | LearnLoop RAG 模块 | 数值断言 pytest（4 用例）+ 实现须含对数（防硬编码） |
| t03_wilson_ci | 实现 | LearnLoop eval/stats 模块 | 教科书值断言（容差 0.01）+ 边界（p=0/1、n=0）+ 单调性 |
| t04_prefix_sum | 算法 | 信竞背景（线段树） | 正确性（随机对照暴力）+ 性能（20000 元素 10000 次混合作询 <5s，O(n) 暴力约 30s） |

## 主结果

| 模型 | pass rate | 95% CI (Wilson) | pass@3 | pass^3 |
|------|-----------|-----------------|--------|--------|
| glm-5.3-flash | **12/12 = 100%** | [75.7%, 100%] | 1.000 | **1.000** |
| deepseek-flash | 11/12 = 91.7% | [64.6%, 98.5%] | 1.000 | 0.750 |
| kimi-for-coding | 6/12 = 50.0% | [25.4%, 74.6%] | 0.909 | 0.000 |

- **能力 vs 可靠性分离**（pass@3 高 / pass^3 低 = 会做但不稳定）：kimi 能力分 pass@3=0.909 但三次全过的任务数为 0——单次可用性差；deepseek 仅 t02 失手一次；glm-5.3-flash 全绿。
- 小样本提醒：n=12/模型，CI 宽，此为 harness 验证性 pilot 而非结论性榜单。

## 失败归因（trajectory 分析）

- **kimi × t01**：两次输出自由文本而非 `TOOL:` 格式（格式遵循）；step5 空回复触发 API 400 中断 rollout——属 **RUN_ERROR（基础设施/协议）**，非任务能力失败。该 case 提示 harness 需对空回复做防御，v2.1 修复。
- **kimi × t04 两次失败**：线段树实现正确但性能超时 / 或边界处理错——**能力失败**，符合其定位（长上下文编码强、小模型算法弱）。
- **deepseek × t02 一次失败**：数值精度用 float 直除——接近能力失败边缘，temperature=1 下的正常波动。

## Harness 设计要点（实测验证过的）

1. 固定 agent loop：read/write/edit/bash 四工具 + 单一系统提示，所有模型同 harness——分数反映模型不反映脚手架
2. 任务三件套：instruction / verify.sh / solution（solution 只用于 verifier 防抖，打分时永不入工作目录）
3. verifier 防抖工序真实抓到过任务作者的错误：t02 首版测试用例期望值算错（50.5/50.5+1 误当 2.5），参考解×3 全败暴露后修正
4. 统计：pass@k 无偏估计 + pass^k 可靠性 + Wilson 95% CI
5. trajectory 全量落盘 JSON，失败可归因到 step 级

## 下一步（v2.1）

- [ ] 空回复/API 错误防御：重试 + 空内容折叠，RUN_ERROR 与 FAIL 分开统计
- [ ] 任务池扩到 20+（加 refactor / 多文件 / 带 lint 约束的任务）
- [ ] 每任务 verifier 增加 PASS_TO_PASS（仓库既有检查）显式分组
- [ ] 成本/步数/token 计入报告
