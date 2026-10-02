"""Profile 标注层测试: RuleBasedAnnotator, 软标注, κ 校验."""

from llm_eval.learning.profiling import (
    RuleBasedAnnotator,
    annotate_samples,
    assess_annotation_quality,
    PROFILE_DESCRIPTIONS,
    ALL_PROFILE_NAMES,
)
from llm_eval.learning.data import PairwiseSample


def test_rule_annotator_coder_keywords():
    ann = RuleBasedAnnotator()
    scores = ann.annotate("请帮我写一个 Python 函数, 处理 JSON 数据, 实现 API 调用")
    # Coder 关键词命中多, 应最高分
    assert scores["coder"] >= 0.7
    # coder 应是该文本的主标签 (即便 writer 也因"写"命中, coder 应更高)
    assert scores["coder"] == max(scores.values())


def test_rule_annotator_writer_keywords():
    ann = RuleBasedAnnotator()
    scores = ann.annotate("帮我写一篇文章, 翻译这段话, 润色文案")
    assert scores["writer"] >= 0.7


def test_rule_annotator_returns_all_profiles():
    ann = RuleBasedAnnotator()
    scores = ann.annotate("任意文本")
    assert set(scores.keys()) == set(ALL_PROFILE_NAMES)
    for v in scores.values():
        assert 0.0 <= v <= 1.0


def test_rule_annotator_fallback_on_no_match():
    """无关键词命中时应给 writer 兜底分 (Arena 多为写作类)."""
    ann = RuleBasedAnnotator()
    scores = ann.annotate("xyz random gibberish 12345")
    assert scores["writer"] > 0  # 兜底
    assert max(scores.values()) > 0


def test_annotate_samples_inplace():
    samples = [
        PairwiseSample(model_a="a", model_b="b", winner="a",
                       task_text="写 Python 代码 debug bug"),
        PairwiseSample(model_a="a", model_b="b", winner="b",
                       task_text="翻译文章"),
    ]
    ann = RuleBasedAnnotator()
    annotate_samples(samples, ann)
    assert samples[0].profile_scores["coder"] >= 0.7
    assert samples[1].profile_scores["writer"] >= 0.7


def test_annotate_skip_already_annotated():
    """已有 profile_scores 的样本应被跳过 (省成本)."""
    samples = [
        PairwiseSample(model_a="a", model_b="b", winner="a",
                       task_text="写代码",
                       profile_scores={"coder": 0.99}),  # 已标注
    ]
    ann = RuleBasedAnnotator()
    annotate_samples(samples, ann)
    # 应保持原值, 不被覆盖
    assert samples[0].profile_scores["coder"] == 0.99


def test_kappa_perfect_agreement():
    """自动标注与人工完全一致时 κ=1."""
    samples = [
        PairwiseSample(model_a="a", model_b="b", winner="a",
                       profile_scores={"coder": 0.9, "writer": 0.1, "planner": 0.1,
                                       "researcher": 0.1, "reviewer": 0.1, "doubter": 0.1}),
        PairwiseSample(model_a="a", model_b="b", winner="a",
                       profile_scores={"coder": 0.1, "writer": 0.9, "planner": 0.1,
                                       "researcher": 0.1, "reviewer": 0.1, "doubter": 0.1}),
    ]
    # 人工标签与 argmax 完全一致
    human = ["coder", "writer"]
    report = assess_annotation_quality(samples, human, n_check=2)
    assert report.cohen_kappa == 1.0
    assert report.agreement_rate == 1.0
    assert report.verdict == "good"


def test_kappa_random_agreement_low():
    """随机标签 κ 应接近 0 或负."""
    samples = []
    for i in range(50):
        profile = "coder" if i % 2 == 0 else "writer"
        ps = {p: (0.9 if p == profile else 0.1) for p in ALL_PROFILE_NAMES}
        samples.append(PairwiseSample(model_a="a", model_b="b", winner="a", profile_scores=ps))
    # 人工标签随机
    import random
    random.seed(42)
    human = [random.choice(["coder", "writer", "planner"]) for _ in range(50)]
    report = assess_annotation_quality(samples, human, n_check=50, seed=42)
    # 随机一致性, κ 应低
    assert report.cohen_kappa < 0.3


def test_profile_descriptions_complete():
    """6 个 profile 都有描述."""
    from llm_eval.models import Profile
    for p in Profile:
        assert p in PROFILE_DESCRIPTIONS
        assert len(PROFILE_DESCRIPTIONS[p]) > 5
