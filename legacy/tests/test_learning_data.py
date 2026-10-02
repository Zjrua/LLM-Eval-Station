"""数据层测试: PairwiseSample, ModelScoreTable, 合成数据, 过滤, JSONL 往返."""

from llm_eval.models import Dimension, Profile
from llm_eval.learning.data import (
    PairwiseSample,
    ModelScoreTable,
    generate_synthetic_data,
    filter_by_profile,
    filter_covered_models,
    train_test_split,
    save_samples_jsonl,
    load_samples_jsonl,
)


def test_pairwise_sample_decisive():
    s = PairwiseSample(model_a="m1", model_b="m2", winner="a")
    assert s.is_decisive()
    s_tie = PairwiseSample(model_a="m1", model_b="m2", winner="tie")
    assert not s_tie.is_decisive()


def test_pairwise_sample_profile_relevance():
    s = PairwiseSample(
        model_a="m1", model_b="m2", winner="a",
        profile_scores={"coder": 0.8, "writer": 0.3},
    )
    assert s.profile_relevance("coder") == 0.8
    assert s.profile_relevance(Profile.WRITER) == 0.3
    assert s.profile_relevance("doubter") == 0.0  # 未标注


def test_model_score_table_common_dimensions():
    table = ModelScoreTable(scores={
        "m1": {Dimension.D1: 80.0, Dimension.D2: 70.0, Dimension.D3: 90.0},
        "m2": {Dimension.D1: 85.0, Dimension.D2: 75.0},
    })
    common = table.common_dimensions("m1", "m2")
    assert Dimension.D1 in common
    assert Dimension.D2 in common
    assert Dimension.D3 not in common  # m2 缺 D3


def test_synthetic_data_shape_and_signal():
    """合成数据应有正确形状且含可学信号."""
    samples, table, gt = generate_synthetic_data(n=500, seed=42, n_models=12)
    assert len(samples) == 500
    assert len(table.models()) == 12
    assert all(d in gt for d in Dimension)
    # 权重和应为 1
    assert abs(sum(gt.values()) - 1.0) < 0.01
    # winner 应只有 a/b (合成数据无 tie)
    winners = {s.winner for s in samples}
    assert winners <= {"a", "b"}
    # a/b 大致均衡 (随机配对, 50/50 附近)
    n_a = sum(1 for s in samples if s.winner == "a")
    assert 200 < n_a < 300


def test_synthetic_data_has_collinearity():
    """合成数据应注入共线性 (D1↔D8 共享 lang 因子)."""
    samples, table, gt = generate_synthetic_data(n=100, seed=42, n_models=20)
    d1_scores = [table.score(m, Dimension.D1) for m in table.models()]
    d8_scores = [table.score(m, Dimension.D8) for m in table.models()]
    # 计算相关系数
    n = len(d1_scores)
    mean1 = sum(d1_scores) / n
    mean8 = sum(d8_scores) / n
    cov = sum((a - mean1) * (b - mean8) for a, b in zip(d1_scores, d8_scores)) / n
    std1 = (sum((a - mean1) ** 2 for a in d1_scores) / n) ** 0.5
    std8 = (sum((b - mean8) ** 2 for b in d8_scores) / n) ** 0.5
    corr = cov / (std1 * std8) if std1 > 0 and std8 > 0 else 0
    # 应正相关 (共享 lang 因子), 阈值放宽因合成噪声
    assert corr > 0.2, f"D1↔D8 相关性 {corr:.3f} 过低, 共线性未注入"


def test_filter_by_profile():
    samples = [
        PairwiseSample(model_a="a", model_b="b", winner="a",
                       profile_scores={"coder": 0.9, "writer": 0.2}),
        PairwiseSample(model_a="a", model_b="b", winner="b",
                       profile_scores={"coder": 0.3, "writer": 0.8}),
    ]
    coders = filter_by_profile(samples, "coder", threshold=0.5)
    assert len(coders) == 1
    assert coders[0].profile_scores["coder"] == 0.9


def test_filter_covered_models():
    table = ModelScoreTable(scores={"m1": {Dimension.D1: 80}, "m2": {Dimension.D1: 70}})
    samples = [
        PairwiseSample(model_a="m1", model_b="m2", winner="a"),
        PairwiseSample(model_a="m1", model_b="m3", winner="a"),  # m3 不在表里
    ]
    covered = filter_covered_models(samples, table)
    assert len(covered) == 1
    assert covered[0].model_b == "m2"


def test_train_test_split_no_overlap():
    samples = [
        PairwiseSample(model_a="a", model_b="b", winner="a", sample_id=str(i))
        for i in range(100)
    ]
    # 加几个 tie, 应被过滤
    samples.append(PairwiseSample(model_a="a", model_b="b", winner="tie"))
    train, test = train_test_split(samples, test_ratio=0.2, seed=42)
    assert len(train) + len(test) == 100  # tie 被排除
    # 无重叠
    train_ids = {s.sample_id for s in train}
    test_ids = {s.sample_id for s in test}
    assert train_ids.isdisjoint(test_ids)
    # 比例正确
    assert abs(len(test) - 20) <= 1


def test_jsonl_roundtrip(tmp_path):
    samples = [
        PairwiseSample(
            model_a="m1", model_b="m2", winner="a",
            profile_scores={"coder": 0.8}, source="synthetic",
            task_text="test task", sample_id="s1",
        ),
        PairwiseSample(model_a="m1", model_b="m3", winner="b", source="arena"),
    ]
    path = tmp_path / "samples.jsonl"
    save_samples_jsonl(samples, path)
    loaded = load_samples_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].model_a == "m1"
    assert loaded[0].profile_scores == {"coder": 0.8}
    assert loaded[1].source == "arena"
