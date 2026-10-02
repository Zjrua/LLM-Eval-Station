"""评估层测试: decision_rule 纯逻辑测试 + (需 numpyro 的端到端测试)."""

from llm_eval.models import Profile
from llm_eval.learning.evaluate import (
    decision_rule,
    ProfileEval,
    LearningEvalSummary,
    LIFT_THRESHOLD_PP,
    CI_WIDTH_THRESHOLD,
)


# ── decision_rule 纯逻辑测试 (不依赖任何 ML 库) ────────────

def test_decision_use_learned():
    """高 lift + 收敛 + CI 不宽 → USE_LEARNED."""
    decision, reason = decision_rule(
        lift_pp=7.0, ci_too_wide=False, converged=True,
    )
    assert decision == "USE_LEARNED"
    assert "7.0pp" in reason or "≥" in reason


def test_decision_fallback_low_lift():
    """lift 不足 → FALLBACK."""
    decision, reason = decision_rule(
        lift_pp=2.0, ci_too_wide=False, converged=True,
    )
    assert decision == "FALLBACK_HANDCRAFT"
    assert "2.0pp" in reason


def test_decision_fallback_not_converged():
    """未收敛 → FALLBACK."""
    decision, reason = decision_rule(
        lift_pp=10.0, ci_too_wide=False, converged=False,
    )
    assert decision == "FALLBACK_HANDCRAFT"
    assert "未收敛" in reason


def test_decision_fallback_ci_too_wide():
    """CI 过宽 → FALLBACK (共线性导致权重不可识别)."""
    decision, reason = decision_rule(
        lift_pp=10.0, ci_too_wide=True, converged=True,
    )
    assert decision == "FALLBACK_HANDCRAFT"
    assert "置信区间" in reason or "CI" in reason


def test_decision_threshold_boundary():
    """lift 正好等于阈值 → USE_LEARNED (>=)."""
    decision, _ = decision_rule(
        lift_pp=LIFT_THRESHOLD_PP, ci_too_wide=False, converged=True,
    )
    assert decision == "USE_LEARNED"


def test_decision_just_below_threshold():
    """lift 刚低于阈值 → FALLBACK."""
    decision, _ = decision_rule(
        lift_pp=LIFT_THRESHOLD_PP - 0.01, ci_too_wide=False, converged=True,
    )
    assert decision == "FALLBACK_HANDCRAFT"


def test_profile_eval_is_use_learned():
    ev = ProfileEval(
        profile=Profile.CODER, n_heldout=100,
        heldout_accuracy_learned=0.72, brier_score_learned=0.21,
        heldout_accuracy_handcraft=0.65, brier_score_handcraft=0.24,
        lift_pp=7.0, passes_threshold=True, ci_too_wide=False,
        decision="USE_LEARNED",
    )
    assert ev.is_use_learned()


def test_summary_markdown_all_fallback():
    """全部 FALLBACK 时报告应有负面结果提示."""
    summary = LearningEvalSummary(
        profiles={
            "coder": ProfileEval(
                profile=Profile.CODER, n_heldout=100,
                heldout_accuracy_learned=0.66, brier_score_learned=0.23,
                heldout_accuracy_handcraft=0.65, brier_score_handcraft=0.24,
                lift_pp=1.0, passes_threshold=False, ci_too_wide=False,
                decision="FALLBACK_HANDCRAFT",
            ),
        },
    )
    md = summary.to_markdown()
    assert "0/1" in md  # 0 个 USE_LEARNED
    assert "负面结果" in md or "不足" in md


def test_summary_markdown_mixed():
    """混合决策时报告应正确计数."""
    summary = LearningEvalSummary(
        profiles={
            "coder": ProfileEval(
                profile=Profile.CODER, n_heldout=100,
                heldout_accuracy_learned=0.72, brier_score_learned=0.21,
                heldout_accuracy_handcraft=0.65, brier_score_handcraft=0.24,
                lift_pp=7.0, passes_threshold=True, ci_too_wide=False,
                decision="USE_LEARNED",
            ),
            "doubter": ProfileEval(
                profile=Profile.DOUBTER, n_heldout=80,
                heldout_accuracy_learned=0.60, brier_score_learned=0.25,
                heldout_accuracy_handcraft=0.62, brier_score_handcraft=0.24,
                lift_pp=-2.0, passes_threshold=False, ci_too_wide=False,
                decision="FALLBACK_HANDCRAFT",
            ),
        },
    )
    assert summary.n_use_learned == 1
    assert summary.n_total == 2
    md = summary.to_markdown()
    assert "✅" in md and "↩️" in md
