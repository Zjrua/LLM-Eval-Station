"""权重学习子包: 用公开 pairwise 偏好数据学习 Profile 维度权重.

技术路径 (对应面试题修正后的方案):
    公开偏好数据 → Profile 软标注 → 因子分析降维 → 线性 Bradley-Terry (NumPyro)
    → 带 95% CI 的可解释权重 → held-out 评估 → 显著超越手工才上线

依赖:
    需要 `pip install -e ".[learning]"` 安装 numpy / scikit-learn / jax / numpyro.
    不安装此 extra 时, 核心包 (scoring/profiles/models) 仍可独立使用.

公开 API:
    data:        PairwiseSample, load_arena_data, generate_synthetic_data
    features:    diagnose_collinearity, extract_latent_factors, add_cross_terms
    model:       bt_model, fit
    evaluate:    ProfileEval, evaluate_profile, decision_rule
    persist:     load_learned_weights, save_learned_weights
"""

__version__ = "0.1.0"
