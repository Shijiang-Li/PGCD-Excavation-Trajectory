"""
多工况 Scene：在一个共享的 model_F/model_v 上，按工况(material, file_index, 起止扰动, 终止姿态)
构造场景，并提供 条件向量 c、决策 τ 的 bounds、J/objectives/components、τ 归一化。

工况 c = [x_s, z_s, x_e, z_e, ξ_e(end_angle), m(material difficulty=kp)]   (6 维)
决策 τ = [m_y, m_z, ξ_s(start_angle), T(t_end)]                            (4 维，对齐 Generate_samples_*)

复用 ../Diffusion_Excavation：config(路径/模型/超参) + objective._bootstrap(注入模型类+chdir+G)。
模型只加载一次（_models 单例），故可廉价地构造成百上千个工况。
"""
import os
import sys
import numpy as np
import torch

from . import config as C            # noqa: E402
from .config import machine_params  # noqa: E402
from . import objective as OBJ       # noqa: E402

# 物料挖掘难度 = get_f_and_v_input 里的 kp
MAT_DIFFICULTY = {"sand": 1, "soil": 1, "small_stone": 2, "stone": 3}
COND_DIM = 6
TAU_DIM = C.DIM   # 4

_MODELS = None


def _load_trusted_model(path):
    try:
        return torch.load(path, map_location=C.DEV, weights_only=False).eval()
    except TypeError:
        return torch.load(path, map_location=C.DEV).eval()


def _models():
    global _MODELS
    if _MODELS is None:
        G = OBJ._bootstrap()                       # 注入模型类到 __main__ + chdir 到 CUBIC + import G
        mF = _load_trusted_model(machine_params.model("model_F"))
        mV = _load_trusted_model(machine_params.model("model_v"))
        _MODELS = (G, mF, mV)
    return _MODELS


_SCENE_CACHE = None


def raw_geometry(G, kne, material, file_index, offset=0):
    """从原始点云 + 挖掘日志计算 (para 料面多项式系数, tsy 起点 y, tey 终点 y)。"""
    cfg = C.MATERIALS[material]
    pc = G.get_point_cloud_fit(f"{C.ORIG}/{cfg['cloud']}/Point_cloud_{file_index}.ply",
                               f"{C.ORIG}/{cfg['pos']}/Point_cloud_position_{file_index}.txt")
    para = G.get_mat_face_func1(pc, offset, degree=4)
    truef = f"{C.EXC}/{cfg['truedir']}/{cfg['trueprefix']}{file_index}.csv"
    vdis = np.loadtxt(open(truef, "rb"), delimiter=",", skiprows=1, usecols=[13, 14, 15])
    tsy = kne.kine_qdTowz(90, vdis[0, 0], vdis[0, 1], vdis[0, 2])[1]
    tey = kne.kine_qdTowz(90, vdis[-1, 0], vdis[-1, 1], vdis[-1, 2])[1]
    return para, tsy, tey


def _scene_cache():
    """data/scene_cache.npz（scripts/build_scene_cache.py 生成）：{"mat#fi": (para, tsy, tey)}。"""
    global _SCENE_CACHE
    if _SCENE_CACHE is None:
        _SCENE_CACHE = {}
        if os.path.exists(C.SCENE_CACHE):
            d = np.load(C.SCENE_CACHE)
            for k, p, s, e in zip(d["keys"], d["para"], d["tsy"], d["tey"]):
                _SCENE_CACHE[str(k)] = (np.asarray(p, float), float(s), float(e))
    return _SCENE_CACHE


def scene_geometry(G, material, file_index, offset=0):
    """优先读预计算缓存（无需原始点云）；缓存缺失或 offset≠0 时回退到原始数据。"""
    hit = _scene_cache().get(f"{material}#{int(file_index)}") if offset == 0 else None
    if hit is not None and not C.USE_RAW_SCENES:
        return hit[0].copy(), hit[1], hit[2]
    return raw_geometry(G, G.Kinematic(), material, file_index, offset)


class Scene:
    """单个工况的场景 + 目标。"""

    def __init__(self, material="sand", file_index=1, rs=0.0, re=0.0, offset=0, end_angle=None):
        G, mF, mV = _models()
        self.G, self.model_F, self.model_v = G, mF, mV
        self.material = material
        self.file_index = file_index
        self.end_angle = float(C.END_ANGLE if end_angle is None else end_angle)
        self.para, tsy, tey = scene_geometry(G, material, file_index, offset)
        tsy, tey = tsy + rs, tey + re
        self.start_pos, self.end_pos = G.get_start_end_pos(tsy, tey, self.para)
        sp, ep = self.start_pos, self.end_pos
        self.p1 = np.array([tsy + 100, sp[1]])
        self.p2 = np.array([tsy + 100, 0.5 * (sp[1] + ep[1]) - 400])
        self.p3 = np.array([tey - 100, 0.5 * (sp[1] + ep[1]) - 400])
        self.p4 = np.array([tey - 100, ep[1]])
        self.lb = np.array([0.5 * (sp[0] + ep[0]) - 100, 0.5 * (sp[1] + ep[1]) - 200, -90, 4.0])
        self.ub = np.array([0.5 * (sp[0] + ep[0]) + 100, 0.5 * (sp[1] + ep[1]) - 150, -50, 10.0])
        self.m = MAT_DIFFICULTY[material]

    # ---- 工况向量 ----
    @property
    def condition(self):
        sp, ep = self.start_pos, self.end_pos
        return np.array([sp[0], sp[1], ep[0], ep[1], self.end_angle, float(self.m)], dtype=float)

    # ---- τ 归一化（按本工况 bounds 到 [0,1]）----
    def normalize(self, T):
        return (np.asarray(T, float) - self.lb) / (self.ub - self.lb)

    def denormalize(self, Tn):
        return self.lb + np.asarray(Tn, float) * (self.ub - self.lb)

    # ---- 目标 ----
    def J(self, T):
        T = np.atleast_2d(np.asarray(T, float))
        with torch.no_grad():
            return np.asarray(self.G.objective_function(
                T, model_F=self.model_F, model_v=self.model_v, material=self.material, para=self.para,
                max_energy_power=machine_params.get("constraints")["max_energy_power"],
                max_volume=machine_params.get("constraints")["max_volume"],
                start_pos=self.start_pos, end_pos=self.end_pos, end_angle=self.end_angle,
                point_1=self.p1, point_2=self.p2, point_3=self.p3, point_4=self.p4,
                lambda_t=C.LAMBDA[0], lambda_v=C.LAMBDA[1], lambda_e=C.LAMBDA[2]))

    def J_chunked(self, T, chunk=512):
        T = np.atleast_2d(np.asarray(T, float))
        out = np.empty(len(T))
        for i in range(0, len(T), chunk):
            out[i:i + chunk] = self.J(T[i:i + chunk])
        return out

    def components(self, T):
        """每条 τ 的 (fill 装满体积, energy 能耗, time 时间)。"""
        G = self.G
        x = np.atleast_2d(np.asarray(T, float))
        b = x.shape[0]
        middle = np.zeros((b, 1, 3))
        middle[:, 0, 0] = x[:, 0]; middle[:, 0, 1] = x[:, 1]; middle[:, 0, 2] = x[:, -1] / 2
        t_end = x[:, -1]; start_angle = x[:, -2]
        pos, traj, vel, acc, tao, tao_v, t, cp, dis = G.trajectory_planning_cubic_spline(
            self.start_pos, middle, self.end_pos, start_angle, self.end_angle, t_end)
        fi, vi = G.get_f_and_v_input(self.para, t, traj, tao, self.material, dis, vel)
        with torch.no_grad():
            fill = np.asarray(G.calc_volume(vi, self.model_v)).reshape(-1)
            energy = np.asarray(G.calc_energy_consumption(fi, vel, tao, tao_v, self.model_F, t)).reshape(-1)
        return fill, energy, np.asarray(t_end).reshape(-1)

    def objectives(self, T):
        """多目标 [time, energy, fill_deficit] + feasible 掩码。"""
        fill, energy, time = self.components(T)
        feas = np.isfinite(self.J_chunked(T))
        F = np.column_stack([time, energy, machine_params.get("constraints")["max_volume"] - fill])
        return F, feas

    def objective_callable(self):
        """供 PSO：吃 (pop,4) τ 返回 (pop,) J。"""
        return lambda T: self.J(T)


if __name__ == "__main__":
    import time
    print("building scenes across materials ...")
    for mat, fi in [("sand", 1), ("soil", 1), ("stone", 2), ("small_stone", 3)]:
        t0 = time.time()
        s = Scene(mat, fi)
        c = s.condition
        X = s.lb + (s.ub - s.lb) * np.random.RandomState(0).rand(64, TAU_DIM)
        J = s.J_chunked(X)
        fin = np.isfinite(J)
        jstar = f"{J[fin].min():.4f}" if fin.any() else "n/a(random)"
        print(f"{mat:11s}#{fi}  c={np.round(c,1)}  feasible {fin.sum()}/64  J*={jstar}  ({time.time()-t0:.1f}s)")
    print("scene.py OK")
