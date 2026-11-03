"""多候选生成与筛选：同一工况生成 N 条 τ，按真实多目标 J 选最优 τ*=argmin J。"""
import time
import numpy as np
from . import paper_metrics as PM


def generate_select(diff, scene, c, n=64, guidance=None, eta=0.0, guide_last=40):
    t0 = time.time()
    Tn = diff.sample(c, n=n, guidance=guidance, eta=eta, guide_last=guide_last)
    lat = time.time() - t0
    J = scene.J_chunked(scene.denormalize(Tn))
    f = np.isfinite(J)
    if f.any():
        bi = int(np.argmin(np.where(f, J, np.inf)))
        best_J, best_tau = float(J[bi]), Tn[bi]
    else:
        best_J, best_tau = np.inf, Tn[0]
    return dict(Tn=Tn, J=J, feas=f, best_J=best_J, best_tau=best_tau,
                feas_rate=float(f.mean()), viol_rate=float(1.0 - f.mean()), lat=lat,
                div=PM.pairwise_diversity(Tn[f] if int(f.sum()) > 1 else Tn))
