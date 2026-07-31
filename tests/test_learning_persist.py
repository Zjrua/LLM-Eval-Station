"""持久化层测试: ProfileWeightsArtifact, JSON 往返, ProfileWeights drop-in.

这些测试不依赖 numpy/numpyro (纯 dataclass + JSON).
"""

import json
from llm_eval.models import Dimension, Profile, ProfileWeights
from llm_eval.learning.persist import (
    ProfileWeightsArtifact,
    LearnedWeightsFile,
    save_learned_weights,
    load_learned_weights,
    to_profile_weights,
    load_profile_weights_for_eval,
    build_artifact_from_result,
)


def _make_test_weights():
    """构造测试用的归一化权重 (Σ=1)."""
    w = {d.name: 1.0 / 9 for d in Dimension}  # 等权重, Σ=1
    return w


def test_artifact_normalization_on_load():
    """load 后转 ProfileWeights 应通过 Σ=1 校验."""
    w = _make_test_weights()
    artifact = ProfileWeightsArtifact(
        decision="USE_LEARNED",
        weights=w,
        weights_ci={d: [w[d], w[d]] for d in w},
        handcraft_weights=dict(w),
    )
    pw = to_profile_weights(artifact, Profile.CODER)
    assert isinstance(pw, ProfileWeights)
    assert pw.profile == Profile.CODER
    assert abs(sum(pw.weights.values()) - 1.0) < 0.01


def test_save_load_roundtrip(tmp_path):
    """JSON 往返应保持数据完整."""
    w = _make_test_weights()
    artifacts = {
        Profile.CODER: ProfileWeightsArtifact(
            decision="USE_LEARNED", weights=dict(w),
            weights_ci={d: [w[d] - 0.02, w[d] + 0.02] for d in w},
            handcraft_weights=dict(w),
            lift_pp=6.5, converged=True, n_train=1000, n_heldout=200,
            heldout_accuracy_learned=0.72, heldout_accuracy_handcraft=0.65,
        ),
        Profile.DOUBTER: ProfileWeightsArtifact(
            decision="FALLBACK_HANDCRAFT", weights=dict(w),
            weights_ci={d: [w[d], w[d]] for d in w},
            handcraft_weights=dict(w),
            reason="样本不足", lift_pp=1.2,
        ),
    }
    path = tmp_path / "weights.json"
    save_learned_weights(artifacts, path, data_source="test")

    loaded = load_learned_weights(path)
    assert loaded.version == "0.1.0"
    assert loaded.data_source == "test"
    assert "coder" in loaded.profiles
    assert "doubter" in loaded.profiles
    assert loaded.profiles["coder"].decision == "USE_LEARNED"
    assert loaded.profiles["coder"].lift_pp == 6.5
    assert loaded.profiles["doubter"].decision == "FALLBACK_HANDCRAFT"
    assert loaded.n_use_learned == 1


def test_load_profile_weights_auto_mode(tmp_path):
    """auto 模式应按 decision 混合加载."""
    w = _make_test_weights()
    # coder 用学出值 (稍微改一下区别手工)
    learned_w = dict(w)
    learned_w["D2"] = 0.30  # coder 编程权重更高
    # 重新归一化
    total = sum(learned_w.values())
    learned_w = {k: v / total for k, v in learned_w.items()}

    artifacts = {
        Profile.CODER: ProfileWeightsArtifact(
            decision="USE_LEARNED", weights=learned_w,
            weights_ci={d: [learned_w[d], learned_w[d]] for d in learned_w},
            handcraft_weights=dict(w),
        ),
        Profile.DOUBTER: ProfileWeightsArtifact(
            decision="FALLBACK_HANDCRAFT", weights=dict(w),  # = handcraft
            weights_ci={d: [w[d], w[d]] for d in w},
            handcraft_weights=dict(w),
        ),
    }
    path = tmp_path / "weights.json"
    save_learned_weights(artifacts, path)

    weights = load_profile_weights_for_eval(path, mode="auto")
    # coder 应是学出值 (D2=0.30 归一化后)
    coder_d2 = weights[Profile.CODER].weights[Dimension.D2]
    assert coder_d2 > 0.15  # 学出值偏高
    # doubter 应是手工值
    doubter_d2 = weights[Profile.DOUBTER].weights[Dimension.D2]
    assert abs(doubter_d2 - 1.0 / 9) < 0.01


def test_load_profile_weights_handcraft_mode(tmp_path):
    """handcraft 模式应忽略文件, 全用手工."""
    from llm_eval.profiles import get_profile_weights

    w = _make_test_weights()
    artifacts = {
        Profile.CODER: ProfileWeightsArtifact(
            decision="USE_LEARNED", weights=w,
            weights_ci={d: [w[d], w[d]] for d in w},
            handcraft_weights=dict(w),
        ),
    }
    path = tmp_path / "weights.json"
    save_learned_weights(artifacts, path)

    weights = load_profile_weights_for_eval(path, mode="handcraft")
    handcraft = get_profile_weights(Profile.CODER)
    # 应等于手工值
    assert weights[Profile.CODER].weights[Dimension.D2] == handcraft.weights[Dimension.D2]
