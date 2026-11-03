"""
PC-STOMP（物理约束随机轨迹优化）—— 复现已发表平台方法 (Li et al., 2026) 的精修阶段，作基线。
从 FNN 初始 τ 出发，迭代：对当前 τ 加 K 个高斯扰动 → 真实目标打分(不可行=惩罚) →
按代价做 softmax 概率加权得到更新方向(标准 STOMP 更新)，噪声随迭代退火。返回精修后的单条 τ 与其真实 J。
"FNN → PC-STOMP" 两阶段规划器 = FNN 给初值 + 本函数精修。
"""
import numpy as np


def pc_stomp(scene, tau0, lb, ub, n_iter=12, K=16, sigma=0.10, lam=0.1, anneal=0.7,
             seed=0, stop_when_feasible=True, min_iter=2):
    """从 FNN 初值出发的轻量物理约束精修:达可行即止(已发表两阶段法的精修阶段,非完整优化)。
    默认预算 ~few×K 次评估 → 约 0.5 s/工况;PSO(完整优化)才是最慢的参照。"""
    rng = np.random.RandomState(seed)
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    tau = np.clip(np.asarray(tau0, float).ravel(), lb, ub)
    best_tau, best_J = tau.copy(), float(scene.J(tau[None])[0])
    for it in range(n_iter):
        s = sigma * (1.0 - anneal * it / max(n_iter - 1, 1))          # 噪声退火
        cand = np.clip(tau[None] + s * span * rng.randn(K, len(tau)), lb, ub)
        J = scene.J_chunked(cand)
        fin = np.isfinite(J)
        if fin.any():
            c = np.where(fin, J, J[fin].max() + 0.5)                  # 不可行→比任何可行都差
        else:
            c = np.zeros(K)                                           # 全不可行→等权随机探索
        c = c - c.min()
        w = np.exp(-c / max(lam, 1e-9)); w = w / w.sum()
        tau = np.clip((w[:, None] * cand).sum(0), lb, ub)            # 概率加权更新
        Jt = float(scene.J(tau[None])[0])
        if np.isfinite(Jt) and (not np.isfinite(best_J) or Jt < best_J):
            best_tau, best_J = tau.copy(), Jt
        if stop_when_feasible and np.isfinite(best_J) and it + 1 >= min_iter:
            break                                                     # 已可行,轻量精修即止
    return best_tau, best_J


if __name__ == "__main__":
    from . import scene as SC
    sc = SC.Scene("small_stone", 1)
    rng = np.random.RandomState(0)
    goods = []
    for k in range(8):
        tau0 = sc.lb + (sc.ub - sc.lb) * rng.rand(4)
        t, J = pc_stomp(sc, tau0, sc.lb, sc.ub, seed=k)
        if np.isfinite(J):
            goods.append(J)
    print(f"feasible {len(goods)}/8" + (f"; refined J mean {np.mean(goods):.4f} min {min(goods):.4f}" if goods else ""))
    print("pc_stomp OK")
