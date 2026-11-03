"""评估指标：质量、可行率、解多样性、规划时延。"""
import numpy as np

from . import config as C


def quality(J):
    J = np.asarray(J)
    Jf = J[np.isfinite(J)]
    if len(Jf) == 0:
        return dict(best=np.nan, median=np.nan, mean=np.nan, n=0)
    return dict(best=float(Jf.min()), median=float(np.median(Jf)),
                mean=float(Jf.mean()), n=int(len(Jf)))


def feasibility(J):
    return float(np.isfinite(np.asarray(J)).mean())


def diversity(X, lb, ub, sub=512, seed=0):
    """生成解的多样性：bounds 归一化后平均成对欧氏距离（子采样）。越大越多样。"""
    X = np.asarray(X, dtype=float)
    if len(X) < 2:
        return 0.0
    Xn = (X - lb) / (np.asarray(ub) - np.asarray(lb) + 1e-9)
    if len(Xn) > sub:
        rng = np.random.RandomState(seed)
        Xn = Xn[rng.choice(len(Xn), sub, replace=False)]
    d = np.linalg.norm(Xn[:, None, :] - Xn[None, :, :], axis=-1)
    iu = np.triu_indices(len(Xn), k=1)
    return float(d[iu].mean())


def summarize(name, J, X, lb, ub, latency_s, n_planned):
    J = np.asarray(J)
    X = np.asarray(X, dtype=float)
    fin = np.isfinite(J)
    Xdiv = X[fin] if fin.any() else X
    q = quality(J)
    return dict(method=name, best=q["best"], median=q["median"], mean=q["mean"],
                feasible=feasibility(J), diversity=diversity(Xdiv, lb, ub),
                latency_s=float(latency_s), n_planned=int(n_planned))
