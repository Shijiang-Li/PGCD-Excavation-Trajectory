"""
条件扩散轨迹生成 p_θ(τ | c)：在归一化轨迹参数 τ_norm 上的条件 DDPM。
- 条件 c(6 维工况)与 τ_norm(4 维)各自标准化后输入去噪 MLP。
- 训练采用 filtered-BC 加权(按 J 偏向优质 PSO 解)，或外部权重。
- 采样支持物理引导去噪钩子：τ_{t-1} = DDPM − η·∇_τ J_phy（梯度由 physics_cost 代理给出）。
"""
import numpy as np
import torch
import torch.nn as nn

DEV = "cpu"
T_DIFF = 100
_betas = torch.linspace(1e-4, 0.02, T_DIFF, device=DEV)
_alphas = 1.0 - _betas
_abar = torch.cumprod(_alphas, dim=0)


class CondDenoiser(nn.Module):
    def __init__(self, tau_dim=4, cond_dim=6, h=160, n_freq=8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * torch.pi)
        self.net = nn.Sequential(
            nn.Linear(tau_dim + 2 * n_freq + cond_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, tau_dim),
        )

    def forward(self, x, t, c):
        tn = (t.float() / T_DIFF).view(-1, 1)
        te = torch.cat([torch.sin(tn * self.freqs), torch.cos(tn * self.freqs)], dim=1)
        return self.net(torch.cat([x, te, c], dim=1))


class CondDiffusion:
    def __init__(self, tau_dim=4, cond_dim=6):
        self.tau_dim, self.cond_dim = tau_dim, cond_dim
        self.model = CondDenoiser(tau_dim, cond_dim).to(DEV)
        self.stats = None   # (tm, ts, cm, cs)

    @staticmethod
    def _wbc(J, temp):
        J = np.asarray(J, float)
        tau = max(temp * (J.std() + 1e-9), 1e-9)
        w = np.exp(-(J - J.min()) / tau)
        return w / w.sum()

    def fit(
        self,
        Tau,
        Cond,
        J,
        steps=6000,
        bs=256,
        lr=8e-4,
        bc_temp=0.3,
        weights=None,
        verbose=True,
        log_every=500,
        return_history=False,
    ):
        Tau, Cond, J = np.asarray(Tau, float), np.asarray(Cond, float), np.asarray(J, float)
        tm, ts = Tau.mean(0), Tau.std(0) + 1e-9
        cm, cs = Cond.mean(0), Cond.std(0) + 1e-9
        self.stats = (tm, ts, cm, cs)
        Tn = torch.tensor((Tau - tm) / ts, dtype=torch.float32, device=DEV)
        Cn = torch.tensor((Cond - cm) / cs, dtype=torch.float32, device=DEV)
        w = np.asarray(weights, float) / np.sum(weights) if weights is not None else self._wbc(J, bc_temp)
        w = torch.tensor(w, dtype=torch.float32, device=DEV)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        history = []
        for it in range(steps):
            idx = torch.multinomial(w, bs, replacement=True)
            x0, c = Tn[idx], Cn[idx]
            t = torch.randint(0, T_DIFF, (bs,), device=DEV)
            noise = torch.randn_like(x0)
            ab = _abar[t].view(-1, 1)
            x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
            loss = ((self.model(x_t, t, c) - noise) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            step = it + 1
            should_log = step == 1 or step == steps or (log_every and step % log_every == 0)
            if should_log:
                row = {"step": int(step), "loss": float(loss.item()), "lr": float(opt.param_groups[0]["lr"])}
                if return_history:
                    history.append(row)
                if verbose:
                    print(f"  [cdiff] step {step}/{steps} loss={row['loss']:.6f} lr={row['lr']:.3e}", flush=True)
        if return_history:
            return self, history
        return self

    def sample(self, cond_raw, n=64, guidance=None, eta=0.0, guide_last=40):
        """给定一个工况 cond_raw(6,)，生成 n 条 τ_norm∈[0,1]。guidance(tau_std,c_std)->grad。"""
        tm, ts, cm, cs = self.stats
        c = torch.tensor((np.asarray(cond_raw, float) - cm) / cs, dtype=torch.float32, device=DEV).view(1, -1).repeat(n, 1)
        x = torch.randn(n, self.tau_dim, device=DEV)
        for ti in reversed(range(T_DIFF)):
            t = torch.full((n,), ti, device=DEV, dtype=torch.long)
            with torch.no_grad():
                eps = self.model(x, t, c)
            ab, a, b = _abar[ti], _alphas[ti], _betas[ti]
            mean = (x - b / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)
            if guidance is not None and eta > 0 and ti < guide_last:
                x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
                grad = guidance(x0, c)
                mean = mean - eta * grad
            x = mean + (torch.sqrt(b) * torch.randn_like(x) if ti > 0 else 0.0)
        Tau_norm = x.cpu().numpy() * ts + tm
        return np.clip(Tau_norm, 0.0, 1.0)

    def std_tau(self, Tau):
        tm, ts, _, _ = self.stats
        return (np.asarray(Tau, float) - tm) / ts

    def std_cond(self, Cond):
        _, _, cm, cs = self.stats
        return (np.asarray(Cond, float) - cm) / cs
