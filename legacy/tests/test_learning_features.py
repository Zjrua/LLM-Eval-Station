"""特征工程层测试 (需 numpy + scikit-learn)."""

from conftest import requires_learning
from llm_eval.models import Dimension
from llm_eval.learning.data import generate_synthetic_data, ModelScoreTable
from llm_eval.learning.features import (
    diagnose_collinearity,
    extract_latent_factors,
    add_cross_terms,
    CrossTermSpec,
    DEFAULT_CROSS_TERMS,
    build_feature_matrix,
    project_pairwise_diff,
)


@requires_learning
def test_diagnose_collinearity_severe():
    """合成数据应检出严重共线性."""
    _, table, _ = generate_synthetic_data(n=100, seed=42, n_models=20)
    report = diagnose_collinearity(table)
    assert report.verdict in ("severe", "moderate", "ok")
    # 合成数据注入了共线性, 应至少有中等共线性
    assert len(report.vif) == 9  # 9 个维度
    assert report.condition_number > 0


@requires_learning
def test_diagnose_collinearity_empty_table():
    """空表应返回 no_data."""
    table = ModelScoreTable(scores={})
    report = diagnose_collinearity(table)
    assert report.verdict == "no_data"


@requires_learning
def test_extract_latent_factors_reduces_dims():
    """因子分析应将 9 维降到 4-6 维."""
    _, table, _ = generate_synthetic_data(n=100, seed=42, n_models=30)
    fm = extract_latent_factors(table, min_explained_variance=0.7)
    assert fm is not None
    assert 3 <= fm.n_factors <= 6
    assert len(fm.loading_matrix) == fm.n_factors
    assert len(fm.loading_matrix[0]) == 9  # 9 个维度
    assert fm.explained_variance > 0.5


@requires_learning
def test_extract_latent_factors_insufficient_data():
    """模型数不足时返回 None."""
    table = ModelScoreTable(scores={
        f"m{i}": {Dimension.D1: 80.0 + i, Dimension.D2: 70.0} for i in range(2)
    })
    fm = extract_latent_factors(table)
    assert fm is None  # 仅 2 个模型, 不足


@requires_learning
def test_add_cross_terms():
    """交叉项应正确追加到特征."""
    factor_features = [[0.1, 0.2], [0.3, 0.4]]
    raw_pairs = [
        ({Dimension.D6: 80.0, Dimension.D7: 70.0}, {Dimension.D6: 60.0, Dimension.D7: 65.0}),
        ({Dimension.D6: 90.0, Dimension.D7: 85.0}, {Dimension.D6: 50.0, Dimension.D7: 45.0}),
    ]
    specs = [CrossTermSpec("D6", "D7")]
    combined, names = add_cross_terms(factor_features, specs=specs, raw_scores_pairs=raw_pairs)
    assert len(combined[0]) == 3  # 2 因子 + 1 交叉项
    assert names == ["factor_0", "factor_1", "cross_D6_D7"]
    # 第一对: D6 diff = (80-60)/100 = 0.2, D7 diff = (70-65)/100 = 0.05, 交叉 = 0.01
    assert abs(combined[0][2] - 0.2 * 0.05) < 1e-6


@requires_learning
def test_add_cross_terms_default_specs():
    """默认交叉项应是 3 个有领域意义的组合."""
    assert len(DEFAULT_CROSS_TERMS) == 3


@requires_learning
def test_build_feature_matrix():
    """完整特征矩阵应能从样本构造."""
    from llm_eval.learning.data import generate_synthetic_data, filter_by_profile
    from llm_eval.models import Profile

    samples, table, _ = generate_synthetic_data(n=200, seed=42, n_models=15, profile=Profile.CODER)
    fm = extract_latent_factors(table)
    assert fm is not None
    coders = filter_by_profile(samples, "coder", 0.5)
    features, labels, names = build_feature_matrix(coders, table, fm)
    assert len(features) == len(labels)
    assert len(features) > 0
    # 特征维度 = 因子数 + 3 交叉项
    assert len(features[0]) == fm.n_factors + 3
    assert all(lb in (0, 1) for lb in labels)


@requires_learning
def test_project_pairwise_diff():
    """单样本投影应返回潜因子特征."""
    from llm_eval.models import Profile
    from llm_eval.learning.data import generate_synthetic_data

    samples, table, _ = generate_synthetic_data(n=50, seed=42, n_models=10, profile=Profile.CODER)
    fm = extract_latent_factors(table)
    models = table.models()
    feats, names, raw = project_pairwise_diff(models[0], models[1], table, fm)
    assert len(feats) == fm.n_factors
    assert len(names) == fm.n_factors
