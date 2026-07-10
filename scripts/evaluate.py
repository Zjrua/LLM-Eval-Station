#!/usr/bin/env python3
"""CLI 入口：对大模型进行多维度评测并生成报告.

用法:
    python scripts/evaluate.py                          # 使用内置示例数据
    python scripts/evaluate.py --data my_scores.csv     # 从 CSV 加载自定义数据
    python scripts/evaluate.py --output report.md       # 指定输出文件
    python scripts/evaluate.py --profile coder,writer   # 只评估特定 Profile
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_eval.models import Dimension, ModelInfo, ModelScores, Profile
from llm_eval.profiles import all_profiles, get_profile_weights
from llm_eval.report import generate_report

# ── 内置示例数据 ──────────────────────────────────────
# 来自公开评测平台 BenchLM.ai + 官方文档 (2026-05)

_DEMO_MODELS = [
    ModelScores(
        model=ModelInfo(
            name="GLM-5.1",
            provider="zai (智谱)",
            parameters="7450B (44B active, MoE, 256 experts)",
            context_length="203K",
            is_open_source=True,
            release_date="2026-04-07",
        ),
        scores={
            Dimension.D1: 82.0,
            Dimension.D2: 83.4,
            Dimension.D3: 89.7,
            Dimension.D4: 86.2,
            Dimension.D5: 64.4,
            Dimension.D6: 91.7,
            Dimension.D7: 80.4,
            Dimension.D8: 84.4,
            Dimension.D9: 82.0,
        },
        sources={d: "BenchLM.ai" for d in Dimension},
    ),
    ModelScores(
        model=ModelInfo(
            name="DeepSeek-V4-Pro",
            provider="deepseek",
            parameters="6850B (37B active, MoE)",
            context_length="1M",
            is_open_source=True,
            release_date="2026-03-24",
        ),
        scores={
            Dimension.D1: 89.7,
            Dimension.D2: 89.5,
            Dimension.D3: 92.4,
            Dimension.D4: 90.1,
            Dimension.D5: 51.5,
            Dimension.D6: None,  # 缺失
            Dimension.D7: 76.8,
            Dimension.D8: 75.0,
            Dimension.D9: None,  # 缺失
        },
        sources={d: "BenchLM.ai" for d in Dimension if d not in (Dimension.D6, Dimension.D9)},
    ),
    ModelScores(
        model=ModelInfo(
            name="DeepSeek-V4-Flash",
            provider="deepseek",
            parameters="2840B (13B active, MoE)",
            context_length="1M",
            is_open_source=True,
            release_date="2026-04-24",
        ),
        scores={
            Dimension.D1: 82.1,
            Dimension.D2: 76.8,
            Dimension.D3: 85.6,
            Dimension.D4: None,  # 缺失
            Dimension.D5: 44.7,
            Dimension.D6: None,  # 缺失
            Dimension.D7: 68.5,
            Dimension.D8: 70.0,
            Dimension.D9: None,  # 缺失
        },
        sources={d: "BenchLM.ai" for d in Dimension if d not in (Dimension.D4, Dimension.D6, Dimension.D9)},
    ),
    ModelScores(
        model=ModelInfo(
            name="MiniMax M3",
            provider="minimax",
            parameters="~600B (MoE)",
            context_length="1M",
            is_open_source=False,
            release_date="2026",
            notes="图片 OCR 提取数据，部分维度缺数据",
        ),
        scores={
            Dimension.D1: 78.1,
            Dimension.D2: 80.5,
            Dimension.D3: 83.3,
            Dimension.D4: None,
            Dimension.D5: None,
            Dimension.D6: None,
            Dimension.D7: 83.5,
            Dimension.D8: None,
            Dimension.D9: None,
        },
        sources={d: "MiniMax 官方/第三方" for d in Dimension},
    ),
]


def load_from_csv(path: str) -> list[ModelScores]:
    """从 CSV 加载模型分数.

    CSV 格式：
        model,provider,D1,D2,D3,D4,D5,D6,D7,D8,D9
    """
    models = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores = {}
            for i, dim in enumerate(Dimension, 1):
                val = row.get(f"D{i}", "").strip()
                scores[dim] = float(val) if val and val.lower() != "none" else None
            ms = ModelScores(
                model=ModelInfo(
                    name=row["model"],
                    provider=row.get("provider", "unknown"),
                ),
                scores=scores,
            )
            models.append(ms)
    return models


def main():
    parser = argparse.ArgumentParser(
        description="LLM Eval Workbench — 大模型能力评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/evaluate.py
  python scripts/evaluate.py --data data/my_scores.csv
  python scripts/evaluate.py --profile coder,planner
  python scripts/evaluate.py --output report.md
        """,
    )
    parser.add_argument(
        "--data", type=str, help="CSV 格式的模型分数文件路径"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="report.md",
        help="输出报告文件路径 (默认: report.md)"
    )
    parser.add_argument(
        "--profile", "-p", type=str,
        help="逗号分隔的 Profile 列表 (默认: 全部 6 个)"
    )
    parser.add_argument(
        "--format", "-f", choices=["full", "compact"], default="full",
        help="输出格式: full=完整报告, compact=仅排名表格"
    )
    args = parser.parse_args()

    # 加载数据
    if args.data:
        models = load_from_csv(args.data)
    else:
        models = _DEMO_MODELS

    # 选择 Profiles
    if args.profile:
        profile_names = [p.strip() for p in args.profile.split(",")]
        profiles = [get_profile_weights(Profile(p)) for p in profile_names]
    else:
        profiles = all_profiles()

    if args.format == "compact":
        from llm_eval.scoring import cross_profile_ranking
        ranking = cross_profile_ranking(models, profiles)
        for profile_name, results in ranking.items():
            print(f"\n## {profile_name}")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r.scored.model.name}: {r.scored.weighted_score:.1f} "
                      f"(覆盖 {r.scored.coverage:.0%}, {r.scored.data_quality})")
        return

    # 生成完整报告
    report = generate_report(models, profiles)
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"✅ 报告已生成: {args.output}")
    print(f"   模型数: {len(models)}, Profile 数: {len(profiles)}")


if __name__ == "__main__":
    main()
