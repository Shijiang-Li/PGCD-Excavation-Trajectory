"""
PINN 基线：确定性 c → τ_norm，但训练损失中加入**物理信息项**。

  L = MSE(τ_pred, τ_best) + λ_phys · Ĵ(std(τ_pred), std(c))

Ĵ 是与 PG-Diffusion 共用的可微土–工具代理(含可行性惩罚，越低越可行/越省)。
对比意义:物理约束注入的"位置"不同——
  FNN      无物理;
  PINN     物理在**训练损失**(本文件);
  PC-STOMP 物理在**事后优化**;
  PGCD     物理在**采样时引导**(生成式,多候选)。
PINN 仍是确定性单解,故无多样性、单候选。
"""
import numpy as np
import torch

from .fnn_baseline import FNN, predict_fnn   # 复用网络结构与预测

DEV = "cpu"


def train_pinn(C_train, T_train, surrogate, tau_stats, cond_stats,
               lam_phys=0.05, steps=4000, lr=1e-3, verbose=False, log_every=500, return_history=False):
    """C_train: 原始条件(cond_dim); T_train: τ_norm∈[0,1] 回归目标。
    surrogate: 已训练的 CondSurrogate(预测标准化惩罚-J,越低越好,权重冻结)。
    tau_stats=(tm,ts), cond_stats=(cm,cs): 扩散模型的标准化统计(代理输入空间)。
    返回 (model, (cm_in, cs_in)),预测用 predict_fnn。"""
    cm_in, cs_in = C_train.mean(0), C_train.std(0) + 1e-9
    X = torch.tensor((C_train - cm_in) / cs_in, dtype=torch.float32, device=DEV)
    Y = torch.tensor(T_train, dtype=torch.float32, device=DEV)
    Craw = torch.tensor(C_train, dtype=torch.float32, device=DEV)
    tm = torch.tensor(tau_stats[0], dtype=torch.float32, device=DEV)
    ts = torch.tensor(tau_stats[1], dtype=torch.float32, device=DEV)
    cm = torch.tensor(cond_stats[0], dtype=torch.float32, device=DEV)
    cs = torch.tensor(cond_stats[1], dtype=torch.float32, device=DEV)
    Cstd = (Craw - cm) / cs                       # 代理用的标准化条件

    model = FNN(C_train.shape[1], T_train.shape[1]).to(DEV)
    for p in surrogate.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = X.shape[0]; bs = min(64, n)
    history = []
    for it in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        pred = model(X[idx])                       # τ_norm ∈ [0,1]
        mse = ((pred - Y[idx]) ** 2).mean()
        tau_std = (pred - tm) / ts                 # 进入代理的标准化 τ 空间
        phys = surrogate(tau_std, Cstd[idx]).mean()
        loss = mse + lam_phys * phys
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        step = it + 1
        should_log = step == 1 or step == steps or (log_every and step % log_every == 0)
        if should_log:
            row = {
                "step": int(step),
                "loss": float(loss.item()),
                "mse": float(mse.item()),
                "phys": float(phys.item()),
                "lr": float(opt.param_groups[0]["lr"]),
            }
            if return_history:
                history.append(row)
            if verbose:
                print(
                    f"  [pinn] step {step}/{steps} loss={row['loss']:.6f} "
                    f"mse={row['mse']:.6f} phys={row['phys']:.6f} lr={row['lr']:.3e}",
                    flush=True,
                )
    if return_history:
        return model, (cm_in, cs_in), history
    return model, (cm_in, cs_in)


predict_pinn = predict_fnn
