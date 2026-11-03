"""确定性 FNN 基线：c → τ_norm（单解）。对每个训练工况取其最优可行 τ 作回归目标。"""
import numpy as np
import torch
import torch.nn as nn

DEV = "cpu"


class FNN(nn.Module):
    def __init__(self, cond_dim=6, tau_dim=4, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, tau_dim), nn.Sigmoid(),   # τ_norm ∈ [0,1]
        )

    def forward(self, c):
        return self.net(c)


def best_per_condition(data, train_cids):
    """每个训练工况的最优可行 (c, τ_norm)。"""
    J = data["samp_J"]; fin = np.isfinite(J)
    Cs, Ts = [], []
    for cid in train_cids:
        mask = (data["samp_cond"] == cid) & fin
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        bi = idx[np.argmin(J[idx])]
        Cs.append(data["conds"][cid]); Ts.append(data["samp_tau"][bi])
    return np.array(Cs, float), np.array(Ts, float)


def train_fnn(C_train, T_train, steps=4000, lr=1e-3, verbose=True, log_every=500, return_history=False):
    cm, cs = C_train.mean(0), C_train.std(0) + 1e-9
    X = torch.tensor((C_train - cm) / cs, dtype=torch.float32, device=DEV)
    Y = torch.tensor(T_train, dtype=torch.float32, device=DEV)
    model = FNN(C_train.shape[1], T_train.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = X.shape[0]
    bs = min(64, n)
    history = []
    for it in range(steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        step = it + 1
        should_log = step == 1 or step == steps or (log_every and step % log_every == 0)
        if should_log:
            row = {"step": int(step), "loss": float(loss.item()), "lr": float(opt.param_groups[0]["lr"])}
            if return_history:
                history.append(row)
            if verbose:
                print(f"  [fnn] step {step}/{steps} loss={row['loss']:.6f} lr={row['lr']:.3e}", flush=True)
    if return_history:
        return model, (cm, cs), history
    return model, (cm, cs)


def predict_fnn(model, stats, c):
    cm, cs = stats
    c = np.atleast_2d(np.asarray(c, float))
    with torch.no_grad():
        return model(torch.tensor((c - cm) / cs, dtype=torch.float32, device=DEV)).cpu().numpy()
