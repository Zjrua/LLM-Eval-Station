# Legacy — v1 分场景选型系统（已舍弃，2026-10-02）

v1 的 9 维度×6 角色权重矩阵、52 benchmark 聚合、可学习权重（Bradley-Terry）体系整体归档于此，不再维护。

舍弃理由：二级聚合（聚合别人跑的 benchmark 分数）无法回答"这个模型在我的任务上到底行不行"，权重校准依赖的偏好数据与真实 Agent 场景分布不同。v2 转向执行验证型评测（DeepSWE 范式），见 `../harness/` 与 `../docs/v2-design.md`。
