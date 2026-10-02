# LLM Eval Station v2 — Pilot 定稿结果

> 2026-10-02 · 20 任务 × 3 模型 × 3 采样 = 180 rollouts · temperature=1.0 · 走本机 new-api 网关
> harness: `harness/run_eval.py`（固定 agent loop、四工具、独立沙箱目录、verifier 防抖）
> 复现：`python harness/run_eval.py --models kimi-for-coding,glm-5.3-flash,deepseek-flash --tasks tasks --repeats 3`

## 总表

| 模型 | pass rate | 95% CI | pass@3 | pass^3 |
|---|---|---|---|---|
| glm-5.3-flash | **100.0%** (60/60) | [0.94, 1.00] | 1.00 | **1.00** |
| kimi-for-coding | 98.3% (59/60) | [0.91, 1.00] | 1.00 | 0.95 |
| deepseek-flash | 83.3% (50/60) | [0.72, 0.91] | 1.00 | 0.50 |

pass@k（能力上限，k 次采样至少一次通过的无偏估计）与 pass^k（可靠性，k 次全过的比例）分离是本框架的核心口径：三个模型 pass@3 全部 1.00，但 deepseek-flash 单次成功率 83.3%、三连全过仅 50%——能力够、不稳。

## 任务池（20 个，全部改编自真实项目场景）

t01 意图分类 bug 修复 · t02 BM25 IDF 实现 · t03 Wilson 置信区间 · t04 线段树 · t05 硬编码→数据驱动重构 · t06 滑窗分块 overlap bug · t07 Reflexion 判定器 · t08 日志聚合 · t09 带并列排名 · t10 归一化越界 · t11 多文件修 import+加功能 · t12 Markdown 结构校验 · t13 LRU 缓存（O(1) 性能门限） · t14 CSV 榜单 · t15 整除截断 bug · t16 指数退避重试 · t17 多路检索融合 · t18 区间合并 · t19 配置深拷贝合并 · t20 trajectory 摘要

## 失败归因（deepseek-flash 10 次 FAIL）

| 模式 | 次数 | 说明 |
|---|---|---|
| 一条消息多个 TOOL 调用 | 7 | harness 只执行第一个；模型以为环境在响应、实际后续调用从未执行，误判完成后 DONE。协议遵循失败，非任务能力失败 |
| 工具调用格式错误 | 3 | 输出非 `TOOL: 名称({json})` 格式，重格式提示后仍失败 |

kimi-for-coding 唯一失败（t16 backoff r1）：把 `sleep(base ** attempt)` 写成 `sleep(base * attempt)`（线性而非指数）。

## Verifier 防抖记录（任务作者错误也被抓）

- t02：参考解测试期望值数学错误（ln(2.5) vs ln(2)）——参考解×3 全败暴露
- t06：tail 期望 5 字符，滑窗口径实为 4——同上
- t15：设定的"bug"经穷举反例搜索证伪（乘积含 0 项恰好归 0，边界自然正确）——任务前提错误，整体重设计
- t05：verifier 回归检查「不得含旧变量名」误杀合法实现（从 JSON 加载但沿用旧名）——改为行为断言 + 关键词字面量位置检查

另有一次 harness 自身 bug：`--tasks` 相对路径在 sandbox 目录下失效，180 rollout 整体误判——修复为绝对路径，废弃轮次存档 `results/t05fix-PATHBUG/`。

## 设计原则（承 DeepSWE）

1. 任务三件套：instruction.md（人读） + verify.sh（机器判） + solution/（参考解，用于防抖，不给模型）
2. Verifier 防抖：参考解×3 必须全过 + 原始环境必须失败（no-op 必败），防"verifier 太松"
3. 固定 harness：所有被测模型同一 agent loop、同一工具集、同一系统提示——测模型不测 scaffolding
4. 双口径：pass@k（能力）+ pass^k（可靠性）+ Wilson 95% CI
5. RUN_ERROR 与 FAIL 分离：API 故障/格式中断不计为能力失败
6. trajectory 全量落盘，失败逐条归因（协议遵循 vs 任务能力 vs verifier 假阴/假阳）

## 与 v1 的关系

v1（52 benchmark 加权聚合分场景选型）整体归档至 `legacy/`，理由：它测量的是"别人跑的分数"的加权平均（二级聚合），不是第一手执行证据。v2 每个数字都能用一条命令复现。
