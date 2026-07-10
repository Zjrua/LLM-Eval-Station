"""Markdown 推荐报告生成器."""

from datetime import datetime
from io import StringIO

from .models import Dimension, ModelScores, ProfileWeights
from .profiles import WEIGHT_RATIONALES
from .scoring import ScoringResult, cross_profile_ranking


def generate_report(
    model_scores_list: list[ModelScores],
    profiles: list[ProfileWeights],
    title: str = "LLM 模型评估报告",
) -> str:
    """生成完整的 Markdown 评估报告.

    Args:
        model_scores_list: 待评估模型列表
        profiles: Profile 权重配置列表
        title: 报告标题

    Returns:
        Markdown 格式的完整报告字符串
    """
    out = StringIO()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 标题 ──
    out.write(f"# {title}\n\n")
    out.write(f"> 生成时间：{now}\n\n")
    out.write(f"> 候选模型：{'、'.join(m.model.name for m in model_scores_list)}\n\n")

    # ── 一、方法论 ──
    out.write("---\n\n## 一、评估方法论\n\n")
    out.write("### 1.1 九大能力维度\n\n")
    out.write("| 编号 | 维度 | 代表评测集 |\n")
    out.write("|------|------|-----------|\n")
    for dim in Dimension:
        out.write(f"| {dim.name} | {dim.label} | {dim.benchmarks} |\n")

    out.write("\n### 1.2 维度权重矩阵\n\n")
    dim_names = [f"D{i+1}" for i in range(9)]
    out.write("| Profile | " + " | ".join(dim_names) + " |\n")
    out.write("|---------|" + "|".join([":---:"] * 9) + "|\n")
    for pw in profiles:
        cells = [pw.profile.label.split(" — ")[0]]
        for i, dim in enumerate(Dimension):
            w = pw.weights.get(dim, 0)
            cells.append(f"{w:.0%}" if w >= 0.1 else f"{w:.0%}" if w > 0 else "—")
        out.write("| " + " | ".join(cells) + " |\n")

    # ── 二、模型数据 ──
    out.write("\n### 1.3 缺失数据处理策略\n\n")
    out.write(
        "- **跳过归一化**：缺失维度在计算时跳过，权重归一化到已有维度\n"
        "- **覆盖率标注**：每个模型标注数据覆盖率，\(\\ge 90\\%\\) 为完整\n"
        "- **低覆盖警示**：覆盖率 < 30% 的模型标注为「数据不足」\n"
        "- **局限性**：缺高维度可能低估、缺低维度可能高估\n\n"
    )

    out.write("---\n\n## 二、模型原始得分矩阵\n\n")
    out.write("| 维度 | " + " | ".join(m.model.name for m in model_scores_list) + " |\n")
    out.write("|------|" + "|".join([":---:"] * len(model_scores_list)) + "|\n")
    for dim in Dimension:
        cells = [dim.name]
        for ms in model_scores_list:
            s = ms.scores.get(dim)
            cells.append(f"**{s:.1f}**" if s is not None else "❌")
        out.write("| " + " | ".join(cells) + " |\n")

    out.write("\n### 数据覆盖率\n\n")
    out.write("| 模型 | 覆盖率 | 缺失维度 |\n")
    out.write("|------|--------|---------|\n")
    for ms in model_scores_list:
        missing = ", ".join(d.name for d in ms.missing_dimensions) or "无"
        out.write(f"| {ms.model.name} | {ms.coverage:.0%} | {missing} |\n")

    # ── 三、Profile 排名 ──
    out.write("\n---\n\n## 三、Profile 评分排名\n\n")
    ranking = cross_profile_ranking(model_scores_list, profiles)

    for profile_name, results in ranking.items():
        pw = next(p for p in profiles if p.profile.value == profile_name)
        out.write(f"### {pw.profile.label}\n\n")
        out.write(f"> 💡 **权重理由**：{WEIGHT_RATIONALES.get(pw.profile, '')}\n\n")

        out.write("| # | 模型 | 加权分 | 覆盖率 | 数据质量 |\n")
        out.write("|---|------|--------|--------|----------|\n")
        for i, r in enumerate(results, 1):
            prefix = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"  {i}"
            out.write(
                f"| {prefix} | **{r.scored.model.name}** "
                f"| {r.scored.weighted_score:.1f} "
                f"| {r.scored.coverage:.0%} "
                f"| {r.scored.data_quality} |\n"
            )

        if results and results[0].skipped_dimensions:
            skipped = ", ".join(d.name for d in results[0].skipped_dimensions)
            out.write(f"\n> ⚠️ 最佳模型缺失维度：{skipped}\n")
        out.write("\n")

    # ── 四、局限性与建议 ──
    out.write("---\n\n## 四、局限性\n\n")
    out.write(
        "1. **缺失数据偏差**：数据缺失 ≠ 能力差，跳过归一化可能高估或低估模型\n"
        "2. **评测集重叠**：部分评测集跨维度（如 C-Eval 同时计入综合能力和中文理解），可能双重计数\n"
        "3. **时效性**：评测数据来自各模型发布时，未持续追踪更新\n"
        "4. **维度缺失**：当前体系缺少安全性、创造力等维度\n"
        "5. **聚合偏差**：加权平均假设维度间独立，实际存在交互效应\n"
    )

    out.write("\n---\n\n## 五、版本信息\n\n")
    out.write(f"- 报告版本：自动生成于 {now}\n")
    out.write("- 评估引擎：llm-eval-workbench v0.1.0\n")
    out.write(f"- 模型数量：{len(model_scores_list)}\n")
    out.write(f"- Profile 数量：{len(profiles)}\n")

    return out.getvalue()
