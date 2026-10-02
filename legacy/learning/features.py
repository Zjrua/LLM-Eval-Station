"""特征工程层: 共线性诊断 + 因子分析降维 + 交叉项.

对应面试题的修正方案:
    "模型有隐含能力维度" 这个直觉是对的 (D1/D8 因 C-Eval 共线, 背后有潜因子),
    但解法不是 transformer (数据撑不起), 而是**显式因子分析**提取潜变量.

为什么因子分析而非 PCA:
    - PCA 只保证正交和方差最大, 因子无领域含义
    - FactorAnalysis 假设观测 = 潜因子线性组合 + 噪声, 因子可解释
    - 心理测量学百年传统 (Spearman g factor), 适合"能力维度"场景

设计:
    1. diagnose_collinearity  → 输出 VIF/相关矩阵/条件数, 报告必列
    2. extract_latent_factors → 9 维降到 k 维 (k 通常 4-6), 自动选 k
    3. add_cross_terms        → 加 3 个有领域意义的交叉项, 救交互
    4. project_pairwise_diff  → 把 (A, B) 两模型分差投影到潜因子空间, 供 BT 用
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models import Dimension
from .data import ModelScoreTable


# 默认交叉项: 有领域意义的维度交互 (每个交叉项 = 1 个额外可解释参数)
DEFAULT_CROSS_TERMS: list[tuple[Dimension, Dimension]] = [
    (Dimension.D6, Dimension.D7),  # 指令遵循 × 工具调用: 弱指令 + 强工具 = 工具乱用
    (Dimension.D2, Dimension.D7),  # 编程 × 工具调用: Agent 编码场景
    (Dimension.D3, Dimension.D4),  # 数学 × 科学: 推理复合
]


# ── (a) 共线性诊断 ──────────────────────────────────────────


@dataclass
class CollinearityReport:
    """共线性诊断报告."""

    vif: dict[str, float]  # {dim_name: VIF}, VIF>10 = 严重共线
    correlation_matrix: dict[str, dict[str, float]]  # 9×9 相关矩阵
    condition_number: float  # 条件数, >30 = 严重共线
    high_vif_dims: list[str]  # VIF > 10 的维度
    verdict: str  # "ok" | "moderate" | "severe"

    def to_markdown(self) -> str:
        """转 Markdown, 供报告生成器调用."""
        lines = ["### 共线性诊断\n"]
        lines.append(f"- 条件数: {self.condition_number:.1f} "
                     f"({'<30 OK' if self.condition_number < 30 else '≥30 严重共线'})\n")
        lines.append("\n**VIF (方差膨胀因子, >10 为严重共线):**\n")
        lines.append("| 维度 | VIF | 等级 |")
        lines.append("|------|-----|------|")
        for dim, v in sorted(self.vif.items(), key=lambda x: -x[1]):
            grade = "🔴 严重" if v > 10 else "🟡 中等" if v > 5 else "🟢 OK"
            lines.append(f"| {dim} | {v:.2f} | {grade} |")
        if self.high_vif_dims:
            lines.append(f"\n> ⚠️ 高共线维度: {', '.join(self.high_vif_dims)}。"
                         f"这些维度的权重无法单独识别, 因子分析降维可缓解。\n")
        return "\n".join(lines)


def diagnose_collinearity(table: ModelScoreTable) -> CollinearityReport:
    """诊断维度分表的共线性.

    指标:
        - VIF (方差膨胀因子): 每个维度被其他维度线性解释的程度. VIF=1 完全独立,
          VIF>10 严重共线 (该维度权重估计方差膨胀 10 倍)
        - 条件数: 相关矩阵最大/最小特征值的平方根. >30 严重共线
    """
    import numpy as np

    models = table.models()
    if len(models) < 3:
        return _empty_report()

    # 构造 (n_models × 9) 矩阵, 只用所有模型都有的维度
    common = table.common_dimensions(*models)
    if len(common) < 3:
        return _empty_report()

    X = np.array([[table.score(m, d) for d in common] for m in models])
    dim_names = [d.name for d in common]

    # 相关矩阵
    corr = np.corrcoef(X.T)
    corr_dict = {
        dim_names[i]: {dim_names[j]: float(corr[i, j]) for j in range(len(dim_names))}
        for i in range(len(dim_names))
    }

    # VIF: 对每个维度, 用其他维度回归它, VIF = 1/(1-R²)
    vif: dict[str, float] = {}
    for i in range(len(dim_names)):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        if X_others.shape[1] == 0:
            vif[dim_names[i]] = 1.0
            continue
        # 最小二乘 R²
        try:
            X_others_aug = np.column_stack([np.ones(len(X_others)), X_others])
            beta, *_ = np.linalg.lstsq(X_others_aug, y, rcond=None)
            y_pred = X_others_aug @ beta
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r2 = max(0.0, min(0.999, r2))  # 防 ∞
            vif[dim_names[i]] = 1.0 / (1.0 - r2)
        except Exception:  # noqa: BLE001
            vif[dim_names[i]] = float("inf")

    # 条件数
    try:
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = eigvals[eigvals > 1e-10]
        cond = float(np.sqrt(eigvals.max() / eigvals.min())) if len(eigvals) > 0 else float("inf")
    except Exception:  # noqa: BLE001
        cond = float("inf")

    high_vif = [d for d, v in vif.items() if v > 10]
    if cond > 30 or high_vif:
        verdict = "severe" if cond > 30 else "moderate"
    else:
        verdict = "ok"

    return CollinearityReport(
        vif=vif, correlation_matrix=corr_dict,
        condition_number=cond, high_vif_dims=high_vif, verdict=verdict,
    )


def _empty_report() -> CollinearityReport:
    return CollinearityReport(
        vif={}, correlation_matrix={}, condition_number=0.0,
        high_vif_dims=[], verdict="no_data",
    )


# ── (b) 因子分析降维 ────────────────────────────────────────


@dataclass
class FactorModel:
    """拟合好的因子分析模型 (可序列化, 供推断时投影用)."""

    loading_matrix: list[list[float]]  # (k × 9) 因子载荷, 因子→原维度
    n_factors: int
    dims_in_order: list[str]  # 对应 loading_matrix 列的 Dimension.name 顺序
    explained_variance: float  # 累计解释方差比例
    rotation: str  # "varimax" | "none"

    @property
    def n_features(self) -> int:
        """潜因子数 (BT 模型在这么多维上跑)."""
        return self.n_factors


def extract_latent_factors(
    table: ModelScoreTable,
    *,
    n_factors: Optional[int] = None,
    min_explained_variance: float = 0.8,
    max_factors: int = 6,
) -> Optional[FactorModel]:
    """对模型维度分表做因子分析, 提取潜因子.

    Args:
        table: 模型维度分表
        n_factors: 指定因子数; None=自动选 (使解释方差 ≥ min_explained_variance)
        min_explained_variance: 自动选因子数时的目标解释方差 (默认 0.8)
        max_factors: 因子数上限 (9 维最多降到 6, 否则失去降维意义)

    Returns:
        FactorModel 或 None (数据不足时)
    """
    import numpy as np
    from sklearn.decomposition import FactorAnalysis

    models = table.models()
    if len(models) < 5:  # 因子分析需足够样本
        return None

    common = table.common_dimensions(*models)
    if len(common) < 4:
        return None

    X = np.array([[table.score(m, d) for d in common] for m in models])
    dim_names = [d.name for d in common]

    # 标准化 (因子分析对尺度敏感)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_std = (X - mean) / std

    # 自动选因子数: 从大到小试, 选第一个达到目标解释方差的
    if n_factors is None:
        n_factors = _select_n_factors(X_std, min_explained_variance, max_factors)
    n_factors = max(1, min(n_factors, max_factors, len(common) - 1))

    try:
        fa = FactorAnalysis(n_components=n_factors, rotation="varimax")
        fa.fit(X_std)
    except Exception:  # noqa: BLE001
        # varimax 可能失败, 回退无旋转
        fa = FactorAnalysis(n_components=n_factors)
        fa.fit(X_std)

    # loading_matrix: sklearn 给的是 (n_features, n_components) = (9, k)
    # 我们存储为 (k × 9), 即每行是一个因子在各维度的载荷
    loadings = fa.components_.T  # (9, k) → transpose to (k, 9)? check
    # 实际: components_ shape = (n_components, n_features) = (k, 9). 直接用.
    loading_matrix = fa.components_.tolist()  # (k, 9)

    # 估算解释方差 (FactorAnalysis 不直接给, 用手动重构 R² 近似)
    explained = _fa_explained_variance(fa, X_std)

    return FactorModel(
        loading_matrix=loading_matrix,
        n_factors=n_factors,
        dims_in_order=dim_names,
        explained_variance=explained,
        rotation="varimax",
    )


def _fa_reconstruct(fa, X):
    """手动重构 FA: X_recon = transform(X) @ components_ + mean.

    sklearn 的 FactorAnalysis 在新版去掉了 inverse_transform, 需手动实现.
    FA 模型: X = F @ components_ + noise, F = transform(X).
    """
    import numpy as np
    F = fa.transform(X)  # (n, k)
    # components_ shape = (k, n_features); 重构 = F @ components_ + mean
    X_rec = F @ fa.components_
    if hasattr(fa, "mean_") and fa.mean_ is not None:
        X_rec = X_rec + fa.mean_
    return X_rec


def _fa_explained_variance(fa, X_std) -> float:
    """估算 FA 的解释方差 (重构 R²)."""
    import numpy as np
    X_rec = _fa_reconstruct(fa, X_std)
    ss_res = float(np.sum((X_std - X_rec) ** 2))
    ss_tot = float(np.sum(X_std ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _select_n_factors(X_std, min_var: float, max_k: int) -> int:
    """用重构方差选因子数 (类似 PCA 的 elbow, 但基于 FA 重构)."""
    import numpy as np
    from sklearn.decomposition import FactorAnalysis

    n_features = X_std.shape[1]
    best_k = 1
    for k in range(1, min(max_k, n_features) + 1):
        try:
            fa = FactorAnalysis(n_components=k)
            fa.fit(X_std)
            var = _fa_explained_variance(fa, X_std)
            if var >= min_var:
                return k
            best_k = k
        except Exception:  # noqa: BLE001
            continue
    return best_k


# ── (c) 交叉项 ──────────────────────────────────────────────


@dataclass
class CrossTermSpec:
    """交叉项规格 (可序列化)."""

    dim_a: str  # Dimension.name
    dim_b: str
    rationale: str = ""


def add_cross_terms(
    factor_features: list[list[float]],  # 每个样本的潜因子特征 (已投影)
    *,
    specs: Optional[list[CrossTermSpec]] = None,
    raw_scores_pairs: Optional[list[tuple[dict, dict]]] = None,
) -> tuple[list[list[float]], list[str]]:
    """给潜因子特征追加交叉项.

    交叉项在**原维度**空间计算 (D6×D7), 再追加到潜因子特征后.
    需要 raw_scores_pairs 提供原始维度分.

    Args:
        factor_features: 已投影到潜因子的特征 (N × k)
        specs: 交叉项规格; None=用 DEFAULT_CROSS_TERMS
        raw_scores_pairs: 每个样本的 (model_a_scores, model_b_scores) 原始分

    Returns:
        (combined_features, feature_names) — 追加交叉项后的特征矩阵与列名
    """
    if specs is None:
        specs = [
            CrossTermSpec(d_a.name, d_b.name)
            for d_a, d_b in DEFAULT_CROSS_TERMS
        ]

    if raw_scores_pairs is None:
        # 没有原始分, 只返回潜因子特征
        k = len(factor_features[0]) if factor_features else 0
        return factor_features, [f"factor_{i}" for i in range(k)]

    # 构造 dim_name → index 映射 (假设 raw_scores dict 的 key 是 Dimension)
    # 交叉项值 = (A 分差) × (B 分差), 用归一化后的分差
    combined = []
    k = len(factor_features[0]) if factor_features else 0
    base_names = [f"factor_{i}" for i in range(k)]
    cross_names = [f"cross_{s.dim_a}_{s.dim_b}" for s in specs]

    for idx, feats in enumerate(factor_features):
        row = list(feats)
        a_scores, b_scores = raw_scores_pairs[idx]
        for spec in specs:
            dim_a = _dim_from_name(spec.dim_a)
            dim_b = _dim_from_name(spec.dim_b)
            va = a_scores.get(dim_a)
            vb = a_scores.get(dim_b)
            wa = b_scores.get(dim_a)
            wb = b_scores.get(dim_b)
            if None in (va, vb, wa, wb):
                row.append(0.0)  # 缺失维度交叉项置 0
            else:
                # 归一化到 [0,1] 后取分差乘积
                diff_a = (va - wa) / 100.0
                diff_b = (vb - wb) / 100.0
                row.append(diff_a * diff_b)
        combined.append(row)

    return combined, base_names + cross_names


def _dim_from_name(name: str) -> Dimension:
    """从 Dimension.name (如 'D1') 反查枚举."""
    for d in Dimension:
        if d.name == name:
            return d
    raise ValueError(f"未知 Dimension name: {name}")


# ── 投影: 把 pairwise 分差投影到潜因子空间 ─────────────────


def project_pairwise_diff(
    model_a: str,
    model_b: str,
    table: ModelScoreTable,
    factor_model: FactorModel,
) -> tuple[list[float], list[str], Optional[tuple[dict, dict]]]:
    """计算两模型的维度分差, 投影到潜因子空间.

    流程:
        1. 取两模型共有维度的分数
        2. 标准化 (用拟合时的均值/方差 — 这里简化为 /100)
        3. 计算 diff = score_a - score_b (per dim)
        4. 投影: factor_diff = diff_std @ loading_matrix.T

    Returns:
        (factor_diff_features, factor_names, raw_scores_pair)
        raw_scores_pair 用于后续交叉项计算
    """
    import numpy as np

    common = [d for d in (Dimension[d] for d in factor_model.dims_in_order)
              if table.score(model_a, d) is not None
              and table.score(model_b, d) is not None]

    if len(common) < 2:
        return [], [], None

    # 按 factor_model 的维度顺序取分差
    diff_vec = []
    raw_a, raw_b = {}, {}
    for d_name in factor_model.dims_in_order:
        d = Dimension[d_name]
        sa = table.score(model_a, d)
        sb = table.score(model_b, d)
        if sa is None or sb is None:
            diff_vec.append(0.0)  # 缺失维度分差为 0
        else:
            diff_vec.append((sa - sb) / 100.0)  # 归一化
            raw_a[d] = sa
            raw_b[d] = sb

    # 投影: loading_matrix 是 (k × 9), diff_vec 是 (9,)
    loadings = np.array(factor_model.loading_matrix)  # (k, 9)
    diff_arr = np.array(diff_vec)  # (9,)
    factor_diff = (loadings @ diff_arr).tolist()  # (k,)

    factor_names = [f"factor_{i}" for i in range(factor_model.n_factors)]
    raw_pair = (raw_a, raw_b) if raw_a else None
    return factor_diff, factor_names, raw_pair


def build_feature_matrix(
    samples: list,  # list[PairwiseSample]
    table: ModelScoreTable,
    factor_model: FactorModel,
    *,
    cross_term_specs: Optional[list[CrossTermSpec]] = None,
) -> tuple[list[list[float]], list[int], list[str]]:
    """批量构造 BT 模型的特征矩阵.

    Args:
        samples: decisive pairwise 样本
        table: 模型维度分表
        factor_model: 拟合好的因子模型
        cross_term_specs: 交叉项规格

    Returns:
        (features, labels, feature_names)
        - features: (N × (k + n_cross)) 每行是 A vs B 的特征
        - labels: (N,) 1 if A wins, 0 if B wins
        - feature_names: 列名
    """
    factor_features = []
    raw_pairs = []
    labels = []

    for s in samples:
        if not s.is_decisive():
            continue
        feats, _, raw_pair = project_pairwise_diff(s.model_a, s.model_b, table, factor_model)
        if not feats:
            continue
        factor_features.append(feats)
        raw_pairs.append(raw_pair)
        labels.append(1 if s.winner == "a" else 0)

    if not factor_features:
        return [], [], []

    # 加交叉项
    specs = cross_term_specs or [CrossTermSpec(d_a.name, d_b.name) for d_a, d_b in DEFAULT_CROSS_TERMS]
    raw_pairs_for_cross = [p if p is not None else ({}, {}) for p in raw_pairs]
    combined, names = add_cross_terms(
        factor_features, specs=specs, raw_scores_pairs=raw_pairs_for_cross,
    )
    return combined, labels, names
