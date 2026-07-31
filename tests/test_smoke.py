"""冒烟测试: 确认包结构与核心导入路径正常 (不依赖 learning extra)."""

from llm_eval.models import Dimension, Profile, ProfileWeights
from llm_eval.profiles import all_profiles, get_profile_weights
from llm_eval.scoring import compute_weighted_score


def test_dimension_count():
    assert len(list(Dimension)) == 9


def test_profile_count():
    assert len(list(Profile)) == 6


def test_handcraft_weights_sum_to_one():
    """所有手工权重矩阵的行和应为 1.0 (ProfileWeights 构造时已校验)."""
    for pw in all_profiles():
        assert abs(sum(pw.weights.values()) - 1.0) < 0.01


def test_learning_subpackage_importable():
    """learning 子包的 __init__ 应能被导入, 但具体符号需 lazy import."""
    import llm_eval.learning as learning_pkg
    assert learning_pkg.__version__ == "0.1.0"


def test_conftest_markers_defined():
    """conftest 提供的跳过标记应存在."""
    from conftest import requires_learning, requires_numpyro
    assert callable(requires_learning)
    assert callable(requires_numpyro)
