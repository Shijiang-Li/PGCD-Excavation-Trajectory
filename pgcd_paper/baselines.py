"""基线：你的 PSO（多目标，每场景在线迭代）+ 随机搜索。返回解、值、墙钟时间（用于摊销对比）。"""
import time
import numpy as np

from . import config as C


def run_pso(obj, pop=50, max_iter=30, n_runs=5):
    """复用 PSO_multi.PSO 在真实 objective 上优化。返回 (best_xs, best_Js, total_time)。"""
    from Trajectory_planning.Cartesian_Space.Cubic_Spline.PSO_multi import PSO
    func = obj.objective_callable()
    best_xs, best_Js = [], []
    t0 = time.time()
    for _ in range(n_runs):
        pso = PSO(func=func, dim=C.DIM, pop=pop, max_iter=max_iter, lb=obj.lb, ub=obj.ub)
        bx, bv, _ = pso.run()
        best_xs.append(np.asarray(bx, dtype=float).ravel())
        best_Js.append(float(np.ravel(bv)[0]))
    dt = time.time() - t0
    return np.array(best_xs), np.array(best_Js), dt


def random_search(obj, n=2048, seed=1):
    t0 = time.time()
    rng = np.random.RandomState(seed)
    X = obj.lb + (obj.ub - obj.lb) * rng.rand(n, C.DIM)
    J = obj.J_chunked(X)
    dt = time.time() - t0
    fin = np.isfinite(J)
    return X[fin], J[fin], dt
