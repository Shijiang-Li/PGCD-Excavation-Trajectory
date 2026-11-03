"""
条件 VAE 生成式基线：p(τ | c)。回答"为何用扩散而非其他生成模型"。
与扩散同样的接口：训练于可行(c, τ_norm)+filtered-BC 加权；推理对一个工况采样 N 个 τ_norm∈[0,1]。
"""
import numpy as np
import torch
import torch.nn as nn

DEV = "cpu"


class CVAE(nn.Module):
    def __init__(self, tau_dim=4, cond_dim=6, z_dim=8, h=128):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(nn.Linear(tau_dim + cond_dim, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU())
        self.mu = nn.Linear(h, z_dim)
        self.lv = nn.Linear(h, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim + cond_dim, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, tau_dim), nn.Sigmoid())   # τ_norm ∈ [0,1]

    def encode(self, tau, c):
        h = self.enc(torch.cat([tau, c], 1))
        return self.mu(h), self.lv(h)

    def decode(self, z, c):
        return self.dec(torch.cat([z, c], 1))


def _wbc(J, temp):
    J = np.asarray(J, float)
    t = max(temp * (J.std() + 1e-9), 1e-9)
    w = np.exp(-(J - J.min()) / t)
    return w / w.sum()


class CVAEModel:
    def __init__(self, tau_dim=4, cond_dim=6, z_dim=8):
        self.net = CVAE(tau_dim, cond_dim, z_dim).to(DEV)
        self.cm = self.cs = None

    def fit(
        self,
        Tau,
        Cond,
        J,
        steps=5000,
        bs=256,
        lr=1e-3,
        bc_temp=0.3,
        beta_kl=0.5,
        verbose=False,
        log_every=500,
        return_history=False,
    ):
        Tau, Cond = np.asarray(Tau, float), np.asarray(Cond, float)
        self.cm, self.cs = Cond.mean(0), Cond.std(0) + 1e-9
        T = torch.tensor(Tau, dtype=torch.float32, device=DEV)
        C = torch.tensor((Cond - self.cm) / self.cs, dtype=torch.float32, device=DEV)
        w = torch.tensor(_wbc(J, bc_temp), dtype=torch.float32, device=DEV)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        n = T.shape[0]
        history = []
        for it in range(steps):
            idx = torch.multinomial(w, bs, replacement=True)
            t, c = T[idx], C[idx]
            mu, lv = self.net.encode(t, c)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            rec = self.net.decode(z, c)
            recon = ((rec - t) ** 2).sum(1).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
            loss = recon + beta_kl * kl
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            step = it + 1
            should_log = step == 1 or step == steps or (log_every and step % log_every == 0)
            if should_log:
                row = {
                    "step": int(step),
                    "loss": float(loss.item()),
                    "recon": float(recon.item()),
                    "kl": float(kl.item()),
                    "lr": float(opt.param_groups[0]["lr"]),
                }
                if return_history:
                    history.append(row)
                if verbose:
                    print(
                        f"  [cvae] step {step}/{steps} loss={row['loss']:.6f} "
                        f"recon={row['recon']:.6f} kl={row['kl']:.6f} lr={row['lr']:.3e}",
                        flush=True,
                    )
        if return_history:
            return self, history
        return self

    def sample(self, cond_raw, n=64):
        c = torch.tensor((np.asarray(cond_raw, float) - self.cm) / self.cs,
                         dtype=torch.float32, device=DEV).view(1, -1).repeat(n, 1)
        with torch.no_grad():
            z = torch.randn(n, self.net.z_dim, device=DEV)
            tau = self.net.decode(z, c).cpu().numpy()
        return np.clip(tau, 0.0, 1.0)
