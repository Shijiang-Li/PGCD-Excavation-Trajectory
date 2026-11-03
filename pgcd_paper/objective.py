"""
目标封装：复用你的 Generate_samples_single_optimization5.objective_function。

- RealObjective(material, file_index)：加载真实 model_F/model_v + 点云场景，提供批量 J(X)。
- 关键工程点：
  1) 须把模型类注入 __main__（整模型 pkl 序列化在 __main__），否则 torch.load 失败；
  2) 须 os.chdir 到 CUBIC（你的脚本用相对路径）。
  二者都在 _bootstrap() 里一次性完成，import 本模块即安全（不触发 PSO，G 有 __main__ 保护）。
"""
import os
import sys
import numpy as np
import torch

from . import config as C
from .config import machine_params

_G = None


def _load_trusted_model(path):
    try:
        return torch.load(path, map_location=C.DEV, weights_only=False).eval()
    except TypeError:
        return torch.load(path, map_location=C.DEV).eval()


def _bootstrap():
    """惰性初始化：设 sys.path、注入模型类到 __main__、chdir、import G。返回 G 模块。"""
    global _G
    if _G is not None:
        return _G
    for p in (C.ROOT, C.CUBIC):
        if p not in sys.path:
            sys.path.insert(0, p)
    import __main__
    from Forward_model_diff.forward_model3_multi_task.model_multi_task_BiGRU_TCN import (
        BiGRU_Block, Chomp1d, TemporalBlock, TemporalConvNet, MLP_Block, BiGRU_TCN_Multi_model)
    from Volume_prediction_model.Volume_model3_no_dis.model_InceptionTimeCBAM import (
        InceptionTimeChannelAttention, InceptionTimeSpatialAttention, InceptionTimeCBAM,
        InceptionModuleCBAM, InceptionTimeRegressionCBAM)
    for cls in (BiGRU_Block, Chomp1d, TemporalBlock, TemporalConvNet, MLP_Block, BiGRU_TCN_Multi_model,
                InceptionTimeChannelAttention, InceptionTimeSpatialAttention, InceptionTimeCBAM,
                InceptionModuleCBAM, InceptionTimeRegressionCBAM):
        setattr(__main__, cls.__name__, cls)
    os.chdir(C.CUBIC)
    import Generate_samples_single_optimization5 as G
    _G = G
    return G


class RealObjective:
    """真实多目标代价 J(X)（最小化，不可行=inf）。X:(batch,4)=[my,mz,start_angle,t_end]。"""

    def __init__(self, material="sand", file_index=1):
        G = _bootstrap()
        self.G = G
        self.material = material
        self.file_index = file_index
        self.model_F = _load_trusted_model(machine_params.model("model_F"))
        self.model_v = _load_trusted_model(machine_params.model("model_v"))

        from .scene import scene_geometry
        self.para, tsy, tey = scene_geometry(G, material, file_index, offset=0)
        self.start_pos, self.end_pos = G.get_start_end_pos(tsy, tey, self.para)
        sp, ep = self.start_pos, self.end_pos
        self.p1 = np.array([tsy + 100, sp[1]])
        self.p2 = np.array([tsy + 100, 0.5 * (sp[1] + ep[1]) - 400])
        self.p3 = np.array([tey - 100, 0.5 * (sp[1] + ep[1]) - 400])
        self.p4 = np.array([tey - 100, ep[1]])
        self.lb = np.array([0.5 * (sp[0] + ep[0]) - 100, 0.5 * (sp[1] + ep[1]) - 200, -90, 4.0])
        self.ub = np.array([0.5 * (sp[0] + ep[0]) + 100, 0.5 * (sp[1] + ep[1]) - 150, -50, 10.0])

    def J(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        with torch.no_grad():
            return np.asarray(self.G.objective_function(
                X, model_F=self.model_F, model_v=self.model_v, material=self.material, para=self.para,
                max_energy_power=machine_params.get("constraints")["max_energy_power"],
                max_volume=machine_params.get("constraints")["max_volume"],
                start_pos=self.start_pos, end_pos=self.end_pos, end_angle=C.END_ANGLE,
                point_1=self.p1, point_2=self.p2, point_3=self.p3, point_4=self.p4,
                lambda_t=C.LAMBDA[0], lambda_v=C.LAMBDA[1], lambda_e=C.LAMBDA[2]))

    def J_chunked(self, X, chunk=None):
        chunk = chunk or C.EVAL_CHUNK
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X))
        for i in range(0, len(X), chunk):
            out[i:i + chunk] = self.J(X[i:i + chunk])
        return out

    def objective_callable(self):
        """返回供 PSO 用的 func：吃 (pop,4) 返回 (pop,)。"""
        return lambda X: self.J(X)

    def components(self, X):
        """每条轨迹的物理分量 (fill 装满体积, energy 能耗, time 总时间)。批量，复刻 objective_function 内部。"""
        G = self.G
        x = np.asarray(X, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        b = x.shape[0]
        middle = np.zeros((b, 1, 3))
        middle[:, 0, 0] = x[:, 0]; middle[:, 0, 1] = x[:, 1]; middle[:, 0, 2] = x[:, -1] / 2
        t_end = x[:, -1]; start_angle = x[:, -2]
        pos, traj, vel, acc, tao, tao_v, t, cp, dis = G.trajectory_planning_cubic_spline(
            self.start_pos, middle, self.end_pos, start_angle, C.END_ANGLE, t_end)
        fi, vi = G.get_f_and_v_input(self.para, t, traj, tao, self.material, dis, vel)
        with torch.no_grad():
            fill = np.asarray(G.calc_volume(vi, self.model_v)).reshape(-1)
            energy = np.asarray(G.calc_energy_consumption(fi, vel, tao, tao_v, self.model_F, t)).reshape(-1)
        return fill, energy, np.asarray(t_end).reshape(-1)

    def objectives(self, X):
        """多目标向量 F=[time, energy, fill_deficit]（三者均最小化）+ feasible 掩码。"""
        fill, energy, time = self.components(X)
        feas = np.isfinite(self.J_chunked(X))
        F = np.column_stack([time, energy, machine_params.get("constraints")["max_volume"] - fill])
        return F, feas


if __name__ == "__main__":
    # 自检：构造目标 + 随机 batch 评估
    import time
    obj = RealObjective("sand", 1)
    print("start_pos", obj.start_pos, "end_pos", obj.end_pos)
    print("lb", obj.lb, "ub", obj.ub)
    X = obj.lb + (obj.ub - obj.lb) * np.random.rand(256, 4)
    t0 = time.time()
    J = obj.J_chunked(X)
    fin = np.isfinite(J)
    print(f"feasible {fin.sum()}/{len(J)}; J min/med {J[fin].min():.4f}/{np.median(J[fin]):.4f}; "
          f"{len(J)/(time.time()-t0):.0f} samples/s")
    print("objective.py OK")
