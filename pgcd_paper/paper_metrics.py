"""论文 6 指标的计算辅助。多样性 D = 2/(N(N-1)) Σ_{i<j} ‖τ_i−τ_j‖²（τ_norm 空间）。"""
import numpy as np


def pairwise_diversity(T):
    T = np.asarray(T, float)
    if len(T) < 2:
        return 0.0
    try:
        from scipy.spatial.distance import pdist
        return float((pdist(T, "sqeuclidean")).mean())
    except Exception:
        s = 0.0; n = 0
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                s += float(((T[i] - T[j]) ** 2).sum()); n += 1
        return s / max(n, 1)


def opt_gap(best_J, pso_J):
    """相对最优差距 (J_method − J_PSO)/|J_PSO|。"""
    if not np.isfinite(best_J) or not np.isfinite(pso_J) or abs(pso_J) < 1e-9:
        return np.nan
    return (best_J - pso_J) / abs(pso_J)
