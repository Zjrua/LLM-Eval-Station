#!/usr/bin/env python3
"""示例：使用 llm-eval-workbench 编程接口进行自定义评估."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_eval.models import Dimension, ModelInfo, ModelScores, Profile
from llm_eval.profiles import get_profile_weights
from llm_eval.scoring import compute_weighted_score, rank_models
from llm_eval.report import generate_report

# ═══════════════════════════════════════════════════════
# 示例 1: 给单个模型打分
# ═══════════════════════════════════════════════════════

print("=" * 60)
print("示例 1: 单个模型 × 单个 Profile 打分")
print("=" * 60)

my_model = ModelScores(
    model=ModelInfo(name="My-Custom-Model", provider="custom"),
    scores={
        Dimension.D1: 85.0,  # 综合能力
        Dimension.D2: 78.0,  # 编程
        Dimension.D3: 90.0,  # 数学推理
        Dimension.D4: None,  # 科学推理（缺失）
        Dimension.D5: 60.0,  # 长文本
        Dimension.D6: 88.0,  # 指令遵循
        Dimension.D7: 72.0,  # 工具调用
        Dimension.D8: 80.0,  # 中文理解
        Dimension.D9: None,  # 多轮对话（缺失）
    },
)

coder_weights = get_profile_weights(Profile.CODER)
result = compute_weighted_score(my_model, coder_weights)

print(f"  模型: {result.scored.model.name}")
print(f"  Profile: {result.scored.profile.value}")
print(f"  加权分: {result.scored.weighted_score:.1f}")
print(f"  覆盖率: {result.scored.coverage:.0%}")
print(f"  缺失维度: {[d.name for d in result.skipped_dimensions]}")
print()

# ═══════════════════════════════════════════════════════
# 示例 2: 多个模型排名
# ═══════════════════════════════════════════════════════

print("=" * 60)
print("示例 2: 多个模型在 Coder Profile 下排名")
print("=" * 60)

model_a = ModelScores(
    model=ModelInfo(name="Model-A", provider="openai"),
    scores={
        Dimension.D1: 88.0, Dimension.D2: 92.0, Dimension.D3: 85.0,
        Dimension.D4: None, Dimension.D5: 70.0, Dimension.D6: 90.0,
        Dimension.D7: 82.0, Dimension.D8: 75.0, Dimension.D9: None,
    },
)
model_b = ModelScores(
    model=ModelInfo(name="Model-B", provider="anthropic"),
    scores={
        Dimension.D1: None, Dimension.D2: 95.0, Dimension.D3: 90.0,
        Dimension.D4: 88.0, Dimension.D5: None, Dimension.D6: 85.0,
        Dimension.D7: 78.0, Dimension.D8: None, Dimension.D9: 80.0,
    },
)
model_c = ModelScores(
    model=ModelInfo(name="Model-C", provider="google"),
    scores={
        Dimension.D1: 90.0, Dimension.D2: 80.0, Dimension.D3: 88.0,
        Dimension.D4: 86.0, Dimension.D5: 75.0, Dimension.D6: 87.0,
        Dimension.D7: 80.0, Dimension.D8: 82.0, Dimension.D9: 78.0,
    },
)

rankings = rank_models([model_a, model_b, model_c], coder_weights)
for i, r in enumerate(rankings, 1):
    print(f"  {i}. {r.scored.model.name}: {r.scored.weighted_score:.1f} "
          f"(覆盖 {r.scored.coverage:.0%})")

print()

# ═══════════════════════════════════════════════════════
# 示例 3: 生成完整报告
# ═══════════════════════════════════════════════════════

print("=" * 60)
print("示例 3: 生成 Markdown 报告")
print("=" * 60)

from llm_eval.profiles import all_profiles

report = generate_report(
    [my_model, model_a, model_b, model_c],
    all_profiles()[:3],  # 只取前 3 个 Profile 做演示
    title="自定义模型评测报告",
)

output_path = Path(__file__).parent / "example_report.md"
output_path.write_text(report, encoding="utf-8")
print(f"  ✅ 报告已生成: {output_path}")
print(f"  📄 大小: {len(report):,} 字符")
