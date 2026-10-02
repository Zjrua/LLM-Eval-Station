"""6 个 Agent Profile 的维度权重矩阵.

每个 Profile 定义了该角色在 9 个能力维度上的权重分配（合计 1.0）。
权重来源于对角色核心任务的系统分析。
"""

from .models import Dimension, Profile, ProfileWeights

# ── 权重矩阵 ──────────────────────────────────────────────
# 行 = Profile, 列 = Dimension

PROFILE_WEIGHT_MATRIX: dict[Profile, dict[Dimension, float]] = {
    Profile.CODER: {
        Dimension.D1: 0.15,   # 综合: 理解跨领域需求
        Dimension.D2: 0.30,   # 编程: ★★★ 核心
        Dimension.D3: 0.05,   # 数学: 算法逻辑
        Dimension.D4: 0.03,   # 科学: 科学计算
        Dimension.D5: 0.01,   # 长文本: 偶尔读长代码
        Dimension.D6: 0.15,   # 指令遵循: API 契约
        Dimension.D7: 0.20,   # 工具调用: ★★ 终端/构建工具
        Dimension.D8: 0.10,   # 中文: 注释/文档
        Dimension.D9: 0.01,   # 多轮: 子任务执行
    },
    Profile.PLANNER: {
        Dimension.D1: 0.25,   # 综合: ★★★ 理解全局
        Dimension.D2: 0.08,   # 编程: 理解技术架构
        Dimension.D3: 0.10,   # 数学: 依赖图分析
        Dimension.D4: 0.05,   # 科学: 方案合理性
        Dimension.D5: 0.02,   # 长文本: 处理任务描述
        Dimension.D6: 0.20,   # 指令遵循: ★★ 解析约束
        Dimension.D7: 0.15,   # 工具调用: 编排子 Agent
        Dimension.D8: 0.10,   # 中文: 理解用户需求
        Dimension.D9: 0.05,   # 多轮: 澄清需求
    },
    Profile.WRITER: {
        Dimension.D1: 0.20,   # 综合: 广泛知识
        Dimension.D2: 0.05,   # 编程: 技术文档
        Dimension.D3: 0.05,   # 数学: 数据准确性
        Dimension.D4: 0.03,   # 科学: fact-check
        Dimension.D5: 0.15,   # 长文本: ★★ 长文档生成
        Dimension.D6: 0.15,   # 指令遵循: 格式规范
        Dimension.D7: 0.02,   # 工具调用: 少量
        Dimension.D8: 0.25,   # 中文: ★★★ 核心输出
        Dimension.D9: 0.10,   # 多轮: 上下文连贯
    },
    Profile.RESEARCHER: {
        Dimension.D1: 0.20,   # 综合: 判断信息质量
        Dimension.D2: 0.02,   # 编程: 爬虫/API
        Dimension.D3: 0.08,   # 数学: 数据分析
        Dimension.D4: 0.10,   # 科学: 研究评估
        Dimension.D5: 0.15,   # 长文本: ★★ 阅读大量资料
        Dimension.D6: 0.05,   # 指令遵循: 搜索策略
        Dimension.D7: 0.25,   # 工具调用: ★★★ 搜索/API/爬虫
        Dimension.D8: 0.10,   # 中文: 中文资料搜索
        Dimension.D9: 0.05,   # 多轮: 逐步深入
    },
    Profile.REVIEWER: {
        Dimension.D1: 0.20,   # 综合: 多角度审视
        Dimension.D2: 0.25,   # 编程: ★★★ 代码审查
        Dimension.D3: 0.10,   # 数学: 逻辑验证
        Dimension.D4: 0.10,   # 科学: 方法论评估
        Dimension.D5: 0.03,   # 长文本: 长输出审查
        Dimension.D6: 0.15,   # 指令遵循: 对照规格
        Dimension.D7: 0.05,   # 工具调用: 验证工具输出
        Dimension.D8: 0.10,   # 中文: 中文质量审查
        Dimension.D9: 0.02,   # 多轮: 一致性检查
    },
    Profile.DOUBTER: {
        Dimension.D1: 0.20,   # 综合: 多角度质疑
        Dimension.D2: 0.05,   # 编程: 代码逻辑漏洞
        Dimension.D3: 0.20,   # 数学: ★★★ 推理严密性
        Dimension.D4: 0.20,   # 科学: ★★★ 方法论审视
        Dimension.D5: 0.04,   # 长文本: 论证完整性
        Dimension.D6: 0.10,   # 指令遵循: 检查遗漏
        Dimension.D7: 0.08,   # 工具调用: 工具链可靠性
        Dimension.D8: 0.10,   # 中文: 语义歧义
        Dimension.D9: 0.03,   # 多轮: 前后矛盾
    },
}


def get_profile_weights(profile: Profile) -> ProfileWeights:
    """获取指定 Profile 的权重配置."""
    if profile not in PROFILE_WEIGHT_MATRIX:
        raise ValueError(f"未知 Profile: {profile}")
    return ProfileWeights(profile=profile, weights=dict(PROFILE_WEIGHT_MATRIX[profile]))


def all_profiles() -> list[ProfileWeights]:
    """获取所有 Profile 的权重配置."""
    return [get_profile_weights(p) for p in Profile]


# ── 权重设计理由（用于报告） ─────────────────────────

WEIGHT_RATIONALES: dict[Profile, str] = {
    Profile.CODER: (
        "编程(30%)是核心产出能力，工具调用(20%)反映 Agent 环境中终端/构建工具的稳定使用需求，"
        "综合能力(15%)+指令遵循(15%)确保理解需求并严格遵循 API 契约。"
    ),
    Profile.PLANNER: (
        "综合能力(25%)是理解任务全局的基础，指令遵循(20%)确保精确解析用户约束，"
        "工具调用(15%)反映编排子 Agent、管理 kanban 依赖链的工作流程。"
    ),
    Profile.WRITER: (
        "中文理解(25%)是中文内容输出的核心质量保证，综合能力(20%)支撑广泛知识面，"
        "长文本(15%)+指令遵循(15%)确保长文档的连贯性和格式规范。"
    ),
    Profile.RESEARCHER: (
        "工具调用(25%)是信息获取的主要手段（搜索、API、爬虫），综合能力(20%)支撑信息质量判断，"
        "长文本(15%)+科学推理(10%)确保能处理和评估大量研究材料。"
    ),
    Profile.REVIEWER: (
        "编程(25%)是代码审查的核心，综合能力(20%)+指令遵循(15%)确保多角度对照需求和规范，"
        "数学/科学推理各 10% 确保对逻辑和方法论的评估能力。"
    ),
    Profile.DOUBTER: (
        "数学推理(20%)+科学推理(20%)是质疑逻辑链和方法论的基石，综合能力(20%)支撑多角度审视，"
        "偏重推理严密性而非产出效率。"
    ),
}
