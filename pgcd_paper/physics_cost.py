"""
物理代价与可微条件代理（物理引导去噪的梯度来源）。

J_phy = λ_F J_F + λ_B J_B + λ_E J_E + λ_C J_C
  J_F 峰值阻力/超载, J_B 满斗率不足, J_E 能耗, J_C 运动学/边界违反。
真实 J_phy 经 scipy 样条/约束/深度模型不可微，故用**可微条件代理** Ĵ(τ_std, c_std) 拟合
（训练目标含不可行惩罚→代理可感知可行边界），采样时用其解析梯度引导。

本文件提供：
- CondSurrogate：复合代理（预测标量 J，含可行性惩罚）。← P2 物理引导用此。
- train_cond_surrogate / make_cond_guidance。
P4 消融再拆 J_C(解析运动学) 与 J_perf(性能代理)。
"""
import numpy as np
import torch
import torch.nn as nn

DEV = "cpu"


class CondSurrogate(nn.Module):
    def __init__(self, tau_dim=4, cond_dim=6, h=160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tau_dim + cond_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1),
        )

    def forward(self, tau, c):
        return self.net(torch.cat([tau, c], dim=1)).squeeze(-1)


def penalize(J, margin=0.5):
    """不可行(J=inf)赋惩罚代价(可行最大 + margin·量程)。"""
    J = np.asarray(J, float)
    fin = np.isfinite(J)
    jmax, jmin = J[fin].max(), J[fin].min()
    pen = jmax + margin * (jmax - jmin)
    Jp = J.copy(); Jp[~fin] = pen
    return Jp


def train_cond_surrogate(Tau_std, Cond_std, J, steps=6000, bs=256, lr=1e-3, verbose=True, log_every=500, return_history=False):
    """J 为(含惩罚的)标量代价；内部标准化。返回 (model, (jm, js))。"""
    X = torch.tensor(np.asarray(Tau_std, np.float32), device=DEV)
    Cc = torch.tensor(np.asarray(Cond_std, np.float32), device=DEV)
    J = np.asarray(J, float)
    jm, js = J.mean(), J.std() + 1e-9
    Y = torch.tensor((J - jm) / js, dtype=torch.float32, device=DEV)
    model = CondSurrogate(X.shape[1], Cc.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = X.shape[0]
    history = []
    for it in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        pred = model(X[idx], Cc[idx])
        loss = ((pred - Y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        step = it + 1
        should_log = step == 1 or step == steps or (log_every and step % log_every == 0)
        if should_log:
            row = {"step": int(step), "loss": float(loss.item()), "lr": float(opt.param_groups[0]["lr"])}
            if return_history:
                history.append(row)
            if verbose:
                print(f"  [csurr] step {step}/{steps} loss={row['loss']:.6f} lr={row['lr']:.3e}", flush=True)
    with torch.no_grad():
        r = model(X, Cc) - Y
        r2 = 1 - (r ** 2).sum().item() / max(((Y - Y.mean()) ** 2).sum().item(), 1e-9)
    if verbose:
        print(f"  [csurr] R^2 = {r2:.3f}")
    if return_history:
        if history:
            history[-1]["r2"] = float(r2)
        return model, (jm, js), history
    return model, (jm, js)


def make_cond_guidance(surrogate):
    """g(tau_std, c_std)->归一化 ∂Ĵ/∂τ_std（标准化 τ 空间）。"""
    def g(tau_std, c_std):
        x = tau_std.detach().requires_grad_(True)
        jh = surrogate(x, c_std).sum()
        grad, = torch.autograd.grad(jh, x)
        return grad / (grad.norm(dim=1, keepdim=True) + 1e-8)
    return g


# ============ 仅可行性(约束)代理：用于消融的"运动学/约束引导" ============
class FeasSurrogate(nn.Module):
    def __init__(self, tau_dim=4, cond_dim=6, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tau_dim + cond_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1),
        )

    def forward(self, tau, c):
        return self.net(torch.cat([tau, c], dim=1)).squeeze(-1)   # logit of P(infeasible)


def train_feas_surrogate(Tau_std, Cond_std, infeasible, steps=5000, bs=256, lr=1e-3, verbose=True, log_every=500, return_history=False):
    """infeasible: 1=不可行, 0=可行。预测 P(infeasible) 的 logit。"""
    X = torch.tensor(np.asarray(Tau_std, np.float32), device=DEV)
    Cc = torch.tensor(np.asarray(Cond_std, np.float32), device=DEV)
    Y = torch.tensor(np.asarray(infeasible, np.float32), device=DEV)
    model = FeasSurrogate(X.shape[1], Cc.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    n = X.shape[0]
    history = []
    for it in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        loss = bce(model(X[idx], Cc[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        step = it + 1
        should_log = step == 1 or step == steps or (log_every and step % log_every == 0)
        if should_log:
            row = {"step": int(step), "bce": float(loss.item()), "lr": float(opt.param_groups[0]["lr"])}
            if return_history:
                history.append(row)
            if verbose:
                print(f"  [feas] step {step}/{steps} bce={row['bce']:.6f} lr={row['lr']:.3e}", flush=True)
    if return_history:
        return model, history
    return model


def make_feas_guidance(feas_surrogate):
    """g(tau_std,c_std)->归一化 ∂P(infeasible)/∂τ_std，沿其负向把样本推入可行域。"""
    def g(tau_std, c_std):
        x = tau_std.detach().requires_grad_(True)
        p = torch.sigmoid(feas_surrogate(x, c_std)).sum()
        grad, = torch.autograd.grad(p, x)
        return grad / (grad.norm(dim=1, keepdim=True) + 1e-8)
    return g
