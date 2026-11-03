"""
条件数据集：按工况(material, file_index)采样大量场景，对每个工况跑 PSO 并**记录种群历史**，
收集 (条件 c, 归一化轨迹参数 τ_norm, 标量目标 J) 三元组。无需重新采集示教数据。

- 留一物料泛化：训练物料 {sand, soil, stone}，测试物料 {small_stone}（未见工况）。
- τ 以各工况 bounds 归一化到 [0,1]，使条件扩散在统一空间学习。
- 缓存为 npz；并存每个工况的 (c, lb, ub, meta) 以便评估时重建 Scene。
"""
import os
import sys
import time
import numpy as np

from . import scene as SC
from . import config as C

TRAIN_MATERIALS = ["sand", "soil", "stone"]
TEST_MATERIAL = "small_stone"
MAT_CODE = {"sand": 0, "soil": 1, "stone": 2, "small_stone": 3}
CODE_MAT = {v: k for k, v in MAT_CODE.items()}
RESULTS = C.RESULTS
os.makedirs(RESULTS, exist_ok=True)


class _Rec:
    def __init__(self, f):
        self.f = f; self.X = []; self.J = []

    def __call__(self, X):
        J = self.f(X)
        self.X.append(np.asarray(X, float)); self.J.append(np.asarray(J, float).reshape(-1))
        return J


def _pso_collect(scene, pop, iters, seed):
    from Trajectory_planning.Cartesian_Space.Cubic_Spline.PSO_multi import PSO
    np.random.seed(seed)
    rec = _Rec(scene.J)
    PSO(func=rec, dim=SC.TAU_DIM, pop=pop, max_iter=iters, lb=scene.lb, ub=scene.ub).run()
    return np.concatenate(rec.X, 0), np.concatenate(rec.J, 0)


def build(n_idx_train=25, n_idx_test=15, pop=40, iters=15, seed0=0, cache=True, smoke=False):
    if smoke:
        n_idx_train, n_idx_test, pop, iters = 2, 2, 30, 8
    tag = f"cond_data_{'smoke' if smoke else f'{n_idx_train}t{n_idx_test}_{pop}x{iters}'}.npz"
    path = os.path.join(RESULTS, tag)
    if cache and os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        print(f"[cond_data] cache {tag}: {len(d['samp_J'])} samples over {len(d['conds'])} conditions")
        return {k: d[k] for k in d.files}

    conds, cond_lb, cond_ub, cond_meta = [], [], [], []
    samp_cond, samp_tau, samp_J = [], [], []
    cid = [0]
    t0 = time.time()

    def add(mat, fi, end_angle, is_test):
        try:
            s = SC.Scene(mat, fi, end_angle=end_angle)
        except Exception as e:
            print(f"   skip {mat}#{fi}: {type(e).__name__}"); return
        X, J = _pso_collect(s, pop, iters, seed0 + cid[0])
        feas = np.isfinite(J)
        if feas.sum() < 5:
            print(f"   skip {mat}#{fi}: only {int(feas.sum())} feasible"); return
        conds.append(s.condition); cond_lb.append(s.lb); cond_ub.append(s.ub)
        cond_meta.append([MAT_CODE[mat], fi, float(end_angle), int(is_test)])
        # 保留全部评估点(含不可行 J=inf)：扩散用可行子集，物理代理用惩罚全集。τ_norm∈[0,1]（PSO 已 clip 到 bounds）
        samp_cond.append(np.full(len(J), cid[0], dtype=int))
        samp_tau.append(s.normalize(X)); samp_J.append(J)
        cid[0] += 1

    for mat in TRAIN_MATERIALS:
        for fi in range(1, n_idx_train + 1):
            add(mat, fi, C.END_ANGLE, False)
        print(f"[cond_data] {mat} done; conditions so far {cid[0]}, {time.time()-t0:.0f}s", flush=True)
    for fi in range(1, n_idx_test + 1):
        add(TEST_MATERIAL, fi, C.END_ANGLE, True)

    out = dict(
        conds=np.array(conds, float), cond_lb=np.array(cond_lb, float), cond_ub=np.array(cond_ub, float),
        cond_meta=np.array(cond_meta, float),
        samp_cond=np.concatenate(samp_cond), samp_tau=np.concatenate(samp_tau, 0),
        samp_J=np.concatenate(samp_J),
    )
    n_test = int(out["cond_meta"][:, 3].sum())
    print(f"[cond_data] {len(out['conds'])} conditions ({n_test} test), {len(out['samp_J'])} samples, {time.time()-t0:.0f}s")
    if cache:
        np.savez(path, **out)
    return out


ALL_MATERIALS = ["sand", "soil", "stone", "small_stone"]


def build_loo(n_idx=20, pop=40, iters=15, seed0=100, cache=True):
    """均衡 4 物料数据集（每物料 n_idx 个工况），供留一物料(leave-one-out)实验在实验期按物料划分。"""
    tag = f"cond_data_loo_{n_idx}_{pop}x{iters}.npz"
    path = os.path.join(RESULTS, tag)
    if cache and os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        print(f"[loo-data] cache {tag}: {len(d['samp_J'])} samples over {len(d['conds'])} conditions")
        return {k: d[k] for k in d.files}

    conds, cond_lb, cond_ub, cond_meta = [], [], [], []
    samp_cond, samp_tau, samp_J = [], [], []
    cid = [0]; t0 = time.time()

    def add(mat, fi):
        try:
            s = SC.Scene(mat, fi, end_angle=C.END_ANGLE)
        except Exception as e:
            print(f"   skip {mat}#{fi}: {type(e).__name__}"); return
        X, J = _pso_collect(s, pop, iters, seed0 + cid[0])
        if np.isfinite(J).sum() < 5:
            print(f"   skip {mat}#{fi}: few feasible"); return
        conds.append(s.condition); cond_lb.append(s.lb); cond_ub.append(s.ub)
        cond_meta.append([MAT_CODE[mat], fi, float(C.END_ANGLE), 0])
        samp_cond.append(np.full(len(J), cid[0], dtype=int)); samp_tau.append(s.normalize(X)); samp_J.append(J)
        cid[0] += 1

    for mat in ALL_MATERIALS:
        for fi in range(1, n_idx + 1):
            add(mat, fi)
        print(f"[loo-data] {mat} done; conditions {cid[0]}, {time.time()-t0:.0f}s", flush=True)

    out = dict(conds=np.array(conds, float), cond_lb=np.array(cond_lb, float), cond_ub=np.array(cond_ub, float),
               cond_meta=np.array(cond_meta, float), samp_cond=np.concatenate(samp_cond),
               samp_tau=np.concatenate(samp_tau, 0), samp_J=np.concatenate(samp_J))
    print(f"[loo-data] {len(out['conds'])} conditions, {len(out['samp_J'])} samples, {time.time()-t0:.0f}s")
    if cache:
        np.savez(path, **out)
    return out


def loo_split(data, holdout_material):
    """留出 holdout_material：返回 (train_cond_idx, test_cond_idx)。"""
    code = MAT_CODE[holdout_material]
    mats = data["cond_meta"][:, 0].astype(int)
    return np.where(mats != code)[0], np.where(mats == code)[0]


def split_iood(data, ood_material="soil", train_frac=0.7, val_frac=0.1, seed=0):
    """In-distribution / out-of-distribution split for the fixed-split protocol.

    Three materials are in-distribution and split per-material into train / val /
    familiar-test (default 70/10/20). The fourth material (`ood_material`, default
    soil) is held out entirely as the unfamiliar (extrapolation) test set.
    Returns dict of condition-index arrays: train, val, fam_test, ood_test.
    """
    mats = data["cond_meta"][:, 0].astype(int)
    ood_code = MAT_CODE[ood_material]
    rng = np.random.RandomState(seed)
    train, val, fam = [], [], []
    for m in ALL_MATERIALS:
        if m == ood_material:
            continue
        idx = np.where(mats == MAT_CODE[m])[0].copy()
        rng.shuffle(idx)
        n = len(idx); n_tr = int(round(train_frac * n)); n_va = int(round(val_frac * n))
        train += list(idx[:n_tr]); val += list(idx[n_tr:n_tr + n_va]); fam += list(idx[n_tr + n_va:])
    ood = np.where(mats == ood_code)[0]
    return dict(train=np.array(train, int), val=np.array(val, int),
                fam_test=np.array(fam, int), ood_test=np.array(ood, int))


def reconstruct_scene(meta_row):
    """从 cond_meta 行重建 Scene（用于评估）。"""
    mat = CODE_MAT[int(meta_row[0])]
    return SC.Scene(mat, int(meta_row[1]), end_angle=float(meta_row[2]))


def split(data):
    """返回 (训练样本掩码, 训练条件idx, 测试条件idx)。训练样本=属于训练工况的样本。"""
    is_test_cond = data["cond_meta"][:, 3].astype(bool)
    train_cids = np.where(~is_test_cond)[0]
    test_cids = np.where(is_test_cond)[0]
    train_samp = np.isin(data["samp_cond"], train_cids)
    return train_samp, train_cids, test_cids


if __name__ == "__main__":
    smoke = "--full" not in sys.argv
    d = build(smoke=smoke)
    tr, trc, tec = split(d)
    print(f"samples: {len(d['samp_J'])}  | train-cond {len(trc)} test-cond {len(tec)}  | train-samp {tr.sum()}")
    print(f"cond dim {d['conds'].shape[1]}, tau dim {d['samp_tau'].shape[1]}; "
          f"tau_norm range [{d['samp_tau'].min():.2f},{d['samp_tau'].max():.2f}]")
    print("cond_data.py OK")
