#!/usr/bin/env python3
"""权重学习 CLI: 用 Arena / 合成数据学习 Profile 维度权重.

用法:
    # 合成数据 sanity check (验证 pipeline 能学回 ground truth)
    python scripts/learn_weights.py --source synthetic --n 2000

    # 真实 Arena 数据全流程 (需 huggingface_hub + 标注 LLM API)
    python scripts/learn_weights.py --source arena --annotate llm \\
        --api-key $OPENAI_API_KEY

    # 已标注数据, 只重跑训练 + 评估
    python scripts/learn_weights.py --source arena --skip-annotate \\
        --labeled-data data/arena_labeled.jsonl

    # 只看单个 profile
    python scripts/learn_weights.py --source synthetic --profile coder

输出:
    data/learned_weights.json — 学出权重 + CI + 决策 (供 evaluate.py --weights 消费)
"""

import argparse
import sys
from pathlib import Path

# 复用 evaluate.py 的 sys.path 约定
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_eval.models import Profile
from llm_eval.learning.data import (
    generate_synthetic_data,
    load_arena_data,
    arena_records_to_samples,
    save_samples_jsonl,
    load_samples_jsonl,
)
from llm_eval.learning.profiling import (
    LLMAnnotator,
    RuleBasedAnnotator,
    annotate_samples,
)
from llm_eval.learning.train import train_profile, sanity_check
from llm_eval.learning.evaluate import evaluate_profile
from llm_eval.learning.persist import (
    build_artifact_from_result,
    save_learned_weights,
    get_learned_profiles_summary,
)


def cmd_synthetic(args) -> int:
    """合成数据流程: sanity check + 训练 + 评估."""
    print("=" * 60)
    print("合成数据 sanity check (验证 pipeline 能学回 ground truth)")
    print("=" * 60)

    # 1. sanity check
    sc = sanity_check(
        n_samples=args.n, seed=args.seed,
        profile=Profile(args.profile) if args.profile else Profile.CODER,
        prior_sigma=args.prior_sigma,
    )
    print(sc.to_markdown())
    print()

    if not sc.passed:
        print("⚠️  Sanity check 未通过. pipeline 可能有问题, 请检查:")
        print("  - 因子分析是否正确降维")
        print("  - 先验是否过强 (尝试 --prior-sigma 0.8)")
        print("  - 合成数据信号是否足够 (--n 2000)")
        return 1

    # 2. 全 profile 训练 (合成数据, 多 profile)
    print("=" * 60)
    print("全 Profile 训练 (合成数据)")
    print("=" * 60)

    artifacts = {}
    evals = {}
    for profile in Profile:
        if args.profile and profile.value != args.profile:
            continue
        print(f"\n--- {profile.value} ---")
        samples, table, _ = generate_synthetic_data(
            n=args.n, seed=args.seed + hash(profile.value) % 1000,
            profile=profile,
        )
        result = train_profile(
            samples, table, profile,
            profile_threshold=0.5,
            prior_sigma=args.prior_sigma,
            num_warmup=args.warmup,
            num_samples=args.samples,
            seed=args.seed,
            min_samples=200,
        )

        if result.skipped_reason:
            print(f"  跳过: {result.skipped_reason}")
            continue

        # 评估 (用训练时 split 的 test set)
        from llm_eval.learning.data import train_test_split, filter_covered_models, filter_by_profile
        relevant = filter_by_profile(samples, profile, 0.5)
        covered = filter_covered_models(relevant, table)
        _, test = train_test_split(covered, test_ratio=0.2, seed=args.seed)

        ev = evaluate_profile(result, test, table, result.factor_model)
        print(f"  学出 Acc: {ev.heldout_accuracy_learned:.1%}, "
              f"手工 Acc: {ev.heldout_accuracy_handcraft:.1%}, "
              f"lift: {ev.lift_pp:+.1f}pp")
        print(f"  收敛: {result.fit_result.converged}, "
              f"R-hat: {result.fit_result.r_hat}, ESS: {result.fit_result.ess}")
        print(f"  决策: {ev.decision} — {ev.reason}")

        artifacts[profile] = build_artifact_from_result(result, ev)
        evals[profile] = ev

    # 3. 保存
    out = Path(args.output)
    save_learned_weights(
        artifacts, out,
        data_source=f"synthetic (n={args.n}, seed={args.seed})",
    )
    print(f"\n✅ 权重已保存: {out}")
    print()
    print(get_learned_profiles_summary(out))
    return 0


def cmd_arena(args) -> int:
    """真实 Arena 数据流程."""
    print("=" * 60)
    print("Arena 数据流程")
    print("=" * 60)

    # 1. 加载/下载数据
    samples = []
    if args.labeled_data and Path(args.labeled_data).exists() and args.skip_annotate:
        print(f"加载已标注数据: {args.labeled_data}")
        samples = load_samples_jsonl(args.labeled_data)
    else:
        print(f"下载 Arena 数据 (cache: {args.cache_dir})...")
        try:
            records = load_arena_data(
                cache_dir=args.cache_dir,
                max_samples=args.max_arena_samples,
            )
        except Exception as e:  # noqa: BLE001
            print(f"❌ Arena 数据下载失败: {e}")
            print("提示: 需要 `pip install -e '.[learning]'` 且网络可用")
            return 1
        print(f"  获得 {len(records)} 条原始记录")
        samples = arena_records_to_samples(records)
        print(f"  转换为 {len(samples)} 条 PairwiseSample")

    # 2. 标注
    if not args.skip_annotate:
        print(f"\n标注 Profile 软标注 (annotator: {args.annotate})...")
        if args.annotate == "llm":
            if not args.api_key:
                print("❌ --annotate llm 需要 --api-key")
                return 1
            annotator = LLMAnnotator(
                api_key=args.api_key,
                base_url=args.api_base or "https://api.openai.com/v1",
                model=args.annotate_model,
            )
        else:
            annotator = RuleBasedAnnotator()

        annotate_samples(samples, annotator)
        # 保存标注结果
        labeled_path = Path(args.cache_dir) / "arena_labeled.jsonl"
        save_samples_jsonl(samples, labeled_path)
        print(f"  标注完成, 保存到 {labeled_path}")

    # 3. 加载模型维度分表 (需用户提供, 此处占位)
    table_path = args.model_scores or "data/model_scores_expanded.csv"
    if not Path(table_path).exists():
        print(f"\n⚠️  模型维度分表不存在: {table_path}")
        print("需要统一的 harness 维度分表 (30-50 个模型 × 9 维度).")
        print("请参考 docs/model-evaluation-guide.md 准备此文件, 或用合成数据流程:")
        print("  python scripts/learn_weights.py --source synthetic")
        return 1

    table = _load_score_table(table_path)

    # 4. 训练 + 评估 (复用合成流程的核心逻辑)
    artifacts = {}
    for profile in Profile:
        if args.profile and profile.value != args.profile:
            continue
        print(f"\n--- {profile.value} ---")
        result = train_profile(
            samples, table, profile,
            profile_threshold=args.profile_threshold,
            prior_sigma=args.prior_sigma,
            num_warmup=args.warmup,
            num_samples=args.samples,
            seed=args.seed,
            min_samples=args.min_samples,
        )
        if result.skipped_reason:
            print(f"  跳过: {result.skipped_reason}")
            continue

        from llm_eval.learning.data import train_test_split, filter_by_profile, filter_covered_models
        relevant = filter_by_profile(samples, profile, args.profile_threshold)
        covered = filter_covered_models(relevant, table)
        _, test = train_test_split(covered, test_ratio=0.2, seed=args.seed)
        ev = evaluate_profile(result, test, table, result.factor_model)
        print(f"  lift: {ev.lift_pp:+.1f}pp, 决策: {ev.decision}")
        artifacts[profile] = build_artifact_from_result(result, ev)

    if artifacts:
        out = Path(args.output)
        save_learned_weights(
            artifacts, out,
            data_source=f"lmsys/chatbot_arena_conversations ({len(samples)} samples)",
        )
        print(f"\n✅ 权重已保存: {out}")
        print(get_learned_profiles_summary(out))
    return 0


def _load_score_table(path):
    """从 CSV 加载模型维度分表."""
    import csv
    from llm_eval.models import Dimension
    from llm_eval.learning.data import ModelScoreTable

    scores = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["model"]
            dims = {}
            for i, d in enumerate(Dimension, 1):
                val = row.get(f"D{i}", "").strip()
                if val and val.lower() != "none":
                    dims[d] = float(val)
            scores[name] = dims
    return ModelScoreTable(scores=scores, source=path, harness_version="unknown")


def main():
    parser = argparse.ArgumentParser(
        description="LLM 权重学习: 用偏好数据学习 Profile 维度权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合成数据 sanity check (无需外部依赖)
  python scripts/learn_weights.py --source synthetic --n 2000

  # 真实 Arena 数据 (需标注 LLM)
  python scripts/learn_weights.py --source arena --annotate llm --api-key $KEY

  # 单个 profile
  python scripts/learn_weights.py --source synthetic --profile coder
        """,
    )
    parser.add_argument(
        "--source", choices=["synthetic", "arena"], default="synthetic",
        help="数据源 (默认 synthetic)",
    )
    parser.add_argument("--profile", type=str, help="只训练指定 Profile")
    parser.add_argument("--output", "-o", default="data/learned_weights.json",
                        help="输出权重文件路径")
    # 合成数据参数
    parser.add_argument("--n", type=int, default=2000, help="合成数据样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    # Arena 参数
    parser.add_argument("--cache-dir", default="data/arena_conversations",
                        help="Arena 数据缓存目录")
    parser.add_argument("--max-arena-samples", type=int, default=None,
                        help="Arena 最大样本数 (调试用)")
    parser.add_argument("--labeled-data", default=None,
                        help="已标注数据 JSONL 路径")
    parser.add_argument("--skip-annotate", action="store_true",
                        help="跳过标注 (使用已有标注)")
    parser.add_argument("--annotate", choices=["llm", "rule"], default="rule",
                        help="标注器 (默认 rule 免费测试)")
    parser.add_argument("--annotate-model", default="gpt-4o-mini",
                        help="标注 LLM 模型")
    parser.add_argument("--api-key", default="", help="标注 LLM API key")
    parser.add_argument("--api-base", default="", help="标注 LLM API base URL")
    parser.add_argument("--model-scores", default="data/model_scores_expanded.csv",
                        help="模型维度分表 CSV (Arena 流程必需)")
    # 训练参数
    parser.add_argument("--prior-sigma", type=float, default=0.5,
                        help="先验强度 (0.3=强锚定, 0.5=中, 0.8=弱)")
    parser.add_argument("--warmup", type=int, default=500, help="NUTS warmup 步数")
    parser.add_argument("--samples", type=int, default=1000, help="NUTS 采样数")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="单 profile 最少样本数")
    parser.add_argument("--profile-threshold", type=float, default=0.4,
                        help="Profile 软标注纳入阈值")
    args = parser.parse_args()

    if args.source == "synthetic":
        return cmd_synthetic(args)
    else:
        return cmd_arena(args)


if __name__ == "__main__":
    sys.exit(main())
