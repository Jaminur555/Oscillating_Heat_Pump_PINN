"""Tier 2 (7): GRU black-box baseline (PyTorch, CPU).

Autoregressive dynamical model: GRU cell consumes [t, Tv, dTv/dt, fluid-onehot, Te_prev, Tc_prev]
and predicts (Te, Tc) one step ahead (dt = 25 s model axis).  Trained teacher-forced on the
smoothed training series; at test time it is rolled out autoregressively from the measured
initial state, driven by the measured vessel temperature -- the deep-learning counterpart of
the ODE/PINN rollouts (same information as the MLP/GP features, incl. the measured IC).
"""
import numpy as np
import torch
import torch.nn as nn
from ohp.data import FLUIDS

TSCALE = 1800.0          # same time normalisation as baselines._feats


def _inputs(d, fluid, y_prev):
    """Per-step inputs (N, 8): 6 exogenous features + previous state (N,2)."""
    oh = np.zeros((len(d["t"]), 3)); oh[:, FLUIDS.index(fluid)] = 1
    ex = np.c_[d["t"] / TSCALE, d["Tv"], d["Tv_dot"] * 100.0, oh]
    return np.c_[ex, y_prev]


class _GRU(nn.Module):
    def __init__(self, n_in=8, hidden=64):
        super().__init__()
        self.cell = nn.GRUCell(n_in, hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, h):                      # x (B, n_in) one step
        h = self.cell(x, h)
        return self.head(h), h


def fit_gru(series, seed=0, hidden=64, epochs=1500, lr=5e-3, log=None):
    """series: [((run, fluid), d), ...] -> returns predict(d, fluid) -> (Te, Tc) on d['t']."""
    torch.manual_seed(seed); np.random.seed(seed)

    # exogenous/target arrays; step t>=1 has input (u_t, y_{t-1}) -> target y_t
    seqs = []
    for k, d in series:
        ex = _inputs(d, k[1], np.c_[d["Te"], d["Tc"]])
        seqs.append((ex[1:], np.c_[d["Te"], d["Tc"]][1:]))     # drop t=0 (no predecessor)
    ex_all = np.vstack([s[0] for s in seqs]); y_all = np.vstack([s[1] for s in seqs])
    mx, sx = ex_all.mean(0), ex_all.std(0) + 1e-9
    my, sy = y_all.mean(0), y_all.std(0) + 1e-9
    seqs = [((ex - mx) / sx, (y - my) / sy) for ex, y in seqs]

    n = len(seqs); L = max(len(x) for x, _ in seqs)
    X = torch.zeros(n, L, 8); Y = torch.zeros(n, L, 2); M = torch.zeros(n, L)
    for i, (x, y) in enumerate(seqs):
        X[i, :len(x)] = torch.tensor(x, dtype=torch.float32)
        Y[i, :len(y)] = torch.tensor(y, dtype=torch.float32)
        M[i, :len(x)] = 1.0

    net = _GRU(hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.3)
    for ep in range(epochs):
        h = torch.zeros(n, hidden); loss = 0.0
        for j in range(L):
            if M[:, j].sum() == 0: break
            pred, h = net(X[:, j], h)
            m = M[:, j][:, None]
            loss = loss + (((pred - Y[:, j]) ** 2) * m).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and ep % 100 == 0:
            log(f"    ep {ep:4d}  loss {loss.item():.4f}")

    def predict(d, fluid):
        net.eval()
        oh = np.zeros((len(d["t"]), 3)); oh[:, FLUIDS.index(fluid)] = 1
        ex = np.c_[d["t"] / TSCALE, d["Tv"], d["Tv_dot"] * 100.0, oh]
        with torch.no_grad():
            y = np.array([d["Te"][0], d["Tc"][0]])
            h = torch.zeros(1, hidden)
            outs = [y.copy()]
            for j in range(1, len(d["t"])):
                xj = np.r_[(ex[j] - mx[:6]) / sx[:6], (y - my) / sy]
                p, h = net(torch.tensor(xj, dtype=torch.float32)[None], h)
                y = p[0].numpy() * sy + my
                outs.append(y.copy())
            ys = np.array(outs)
        return ys[:, 0], ys[:, 1]
    return predict
