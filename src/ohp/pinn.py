"""Inverse PINN for a lumped two-node oscillating-heat-pipe (OHP) thermal model (PyTorch).

Physics (per working fluid f, driver = measured vessel temperature Tv(t)):

    dTe/dt = a_f * (Tv - Te) - kappa_f(.) * (Te - Tc)               (evaporator node)
    dTc/dt = r * kappa_f(.) * (Te - Tc) - d * (Tc - Tcool)           (condenser node)

    a_f     [1/s]  vessel->evaporator coupling / evaporator heat capacity   (per fluid)
    kappa_f [1/s]  pipe (evaporator->condenser) conductance / evaporator heat capacity
                   constant per fluid, or  kappa_f * exp(g(f, Te, Te-Tc))  (state dependent)
    r      [-]     C_e / C_c  (shared)
    d      [1/s]   condenser loss rate to the coolant/ambient (shared)
    Tcool  [degC]  condenser sink temperature (shared)

Dimensionless normalised pipe resistance:   R* = R_pipe / R_in = a_f / kappa_f
(it needs no knowledge of the evaporator heat capacity C_e).

The network U(t, series) -> (Te, Tc) is trained on the thermocouple data AND the ODE residual
(computed with autograd); the physical parameters are learned jointly.  This module replaced
the original JAX implementation (retired 2026-10-04); same API: train_pinn, simulate_series,
phys_summary.  RNG streams differ from the JAX version, so same-seed results differ
numerically; reconciliation was statistical (within Tier 1 seed-CIs). float32 throughout.
"""
import numpy as np
import torch
from scipy.interpolate import CubicSpline

torch.set_default_dtype(torch.float32)

T0 = 1800.0                 # time scaling [s]
FLUID_IDX = {"EOHP": 0, "MOHP": 1, "WOHP": 2}
NF = 3


# ----------------------------------------------------------------------------- MLP helpers
def mlp_init(g, sizes, zero_last=False):
    """List of (W, b) torch tensors; g: torch.Generator. Glorot uniform, mirrors mlp_init."""
    ps = []
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        lim = np.sqrt(6.0 / (m + n))
        W = torch.empty(m, n).uniform_(-lim, lim, generator=g)
        if zero_last and i == len(sizes) - 2:
            W = W * 0.0
        ps.append((W, torch.zeros(n)))
    return ps


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = torch.tanh(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def mlp_np(ps):
    return [(W.detach().numpy().copy(), b.detach().numpy().copy()) for W, b in ps]


def mlp_from_np(ps_np):
    return [(torch.tensor(W), torch.tensor(b)) for W, b in ps_np]


# ----------------------------------------------------------------------------- physics (vectorised over points)
def kappa_fn(phys, fl_oh, Te, Tc, state_dependent, kappa_te=False):
    """kappa [1/s]; fl_oh (...,NF), Te/Tc (...,) temperatures in degC.
    kappa_te: kappa_f(Te) = k0_f * exp(beta_f*(Te-50)) instead of constant k0_f."""
    logk = fl_oh @ phys["logK"]
    if kappa_te:
        logk = logk + (fl_oh @ phys["beta"]) * (Te - 50.0)
    if state_dependent:
        x = torch.cat([fl_oh, torch.stack([(Te - 50.0) / 30.0, (Te - Tc - 30.0) / 30.0], dim=-1)], dim=-1)
        logk = logk + mlp(phys["g"], x)[..., 0]
    return torch.exp(logk)


def rhs(phys, fl_oh, Te, Tc, Tv, state_dependent, kappa_te=False):
    a = torch.exp(fl_oh @ phys["loga"])
    kap = kappa_fn(phys, fl_oh, Te, Tc, state_dependent, kappa_te)
    q = kap * (Te - Tc)
    logd = phys["logd"]                                     # scalar (shared d) or (NF,) per-fluid
    d_rate = fl_oh @ logd if np.ndim(logd) else logd
    dTe = a * (Tv - Te) - q
    dTc = torch.exp(phys["logr"]) * q - torch.exp(d_rate) * (Tc - phys["Tcool"])
    return dTe, dTc


def rollout(phys, fl_idx, Tv_half, y0, h, state_dependent, kappa_te=False):
    """RK4, Tv_half sampled every h/2 (len 2N+1). Python loop (evaluate-time only). Returns (N+1,2)."""
    fl_oh = torch.zeros(NF); fl_oh[fl_idx] = 1.0
    tvs = np.stack([Tv_half[0:-1:2], Tv_half[1::2], Tv_half[2::2]], axis=1)

    def f(y, tv):
        dTe, dTc = rhs(phys, fl_oh, y[0], y[1], tv, state_dependent, kappa_te)
        return torch.stack([dTe, dTc])

    with torch.no_grad():
        y = torch.tensor([y0[0], y0[1]], dtype=torch.float32)
        ys = [y.numpy().copy()]
        for t0, tm, t1 in tvs:
            k1 = f(y, torch.tensor(t0, dtype=torch.float32))
            k2 = f(y + 0.5 * h * k1, torch.tensor(tm, dtype=torch.float32))
            k3 = f(y + 0.5 * h * k2, torch.tensor(tm, dtype=torch.float32))
            k4 = f(y + h * k3, torch.tensor(t1, dtype=torch.float32))
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            ys.append(y.numpy().copy())
    return np.array(ys)


def simulate_series(phys, d, fluid, state_dependent, h=1.0, kappa_te=False):
    """Roll the identified ODE over d['t'] using measured Tv(t) + initial state."""
    phys = {k: (mlp_from_np(v) if k == "g" else
                (torch.as_tensor(np.asarray(v), dtype=torch.float32) if not torch.is_tensor(v) else v))
            for k, v in phys.items()}
    t = d["t"]
    n = int(round((t[-1] - t[0]) / h))
    tt = t[0] + np.arange(2 * n + 1) * h / 2.0
    Tv = CubicSpline(t, d["Tv"])(tt)
    ys = rollout(phys, FLUID_IDX[fluid], Tv, (d["Te"][0], d["Tc"][0]), h, state_dependent, kappa_te)
    tg = t[0] + np.arange(n + 1) * h
    return np.interp(t, tg, ys[:, 0]), np.interp(t, tg, ys[:, 1])


# ----------------------------------------------------------------------------- PINN
def build_collocation(series, step_sub=2):
    ts, tv, sid, fl = [], [], [], []
    for s, (k, d) in enumerate(series):
        t = d["t"]
        tc = np.arange(t[0], t[-1] + 1e-9, step_sub * 1.0)
        ts.append(tc); tv.append(CubicSpline(t, d["Tv"])(tc))
        sid.append(np.full(len(tc), s)); fl.append(np.full(len(tc), FLUID_IDX[k[1]]))
    return [np.concatenate(a) for a in (ts, tv, sid, fl)]


def _onehot(idx, n):
    oh = np.zeros((len(idx), n), dtype=np.float32)
    oh[np.arange(len(idx)), idx] = 1.0
    return oh


def train_pinn(series, state_dependent=False, n_steps=6000, seed=0, hidden=(64, 64, 64),
               lam_phys=1.0, verbose=True, init_phys=None, colloc_step=5, d_shared=True, kappa_te=False):
    """series: list of ((run, fluid), data_dict). Returns dict(phys=..., hist=..., ...)."""
    S = len(series)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    netp = [(W.requires_grad_(True), b.requires_grad_(True)) for W, b in mlp_init(g, [1 + S, *hidden, 2])]

    # normalisation constants (from training data only)
    allTe = np.concatenate([d["Te"] for _, d in series]); allTc = np.concatenate([d["Tc"] for _, d in series])
    mu_e, sd_e, mu_c, sd_c = allTe.mean(), allTe.std(), allTc.mean(), allTc.std()
    norm = dict(mu_e=mu_e, sd_e=sd_e, mu_c=mu_c, sd_c=sd_c)

    # measured data tensors
    td = np.concatenate([d["t"] for _, d in series])
    sd_ = np.concatenate([np.full(len(d["t"]), s) for s, (_, d) in enumerate(series)])
    Ted = np.concatenate([d["Te"] for _, d in series])
    Tcd = np.concatenate([d["Tc"] for _, d in series])
    tc_, tvc, sidc, flc = build_collocation(series, step_sub=colloc_step)

    td_t = torch.tensor(td, dtype=torch.float32)
    sdoh_t = torch.tensor(_onehot(sd_, S))
    oh_net = torch.cat([td_t[:, None] / T0, sdoh_t], dim=1)
    Ted_t, Tcd_t = torch.tensor(Ted, dtype=torch.float32), torch.tensor(Tcd, dtype=torch.float32)

    tc_t = torch.tensor(tc_, dtype=torch.float32)
    tc_fl = torch.tensor(_onehot(flc, NF))
    tc_sidoh = torch.tensor(_onehot(sidc, S))
    tc_tv = torch.tensor(tvc, dtype=torch.float32)

    ip = dict(loga=np.log([0.055, 0.032, 0.027]), logK=np.log([0.0048] * 3), logr=np.log(0.03),
              logd=np.log(0.003 if d_shared else [0.003] * 3), Tcool=18.8)
    if init_phys: ip.update(init_phys)
    phys = dict(loga=torch.tensor(ip["loga"], dtype=torch.float32).requires_grad_(True),
                logK=torch.tensor(ip["logK"], dtype=torch.float32).requires_grad_(True),
                logr=torch.tensor(float(ip["logr"])).requires_grad_(True),
                logd=torch.tensor(ip["logd"], dtype=torch.float32).requires_grad_(True),
                Tcool=torch.tensor(float(ip["Tcool"])).requires_grad_(True))
    if kappa_te:
        phys["beta"] = torch.zeros(NF).requires_grad_(True)   # kappa_f(Te)=k0_f*exp(beta_f*(Te-50))
    gp = None
    if state_dependent:
        gp = [(W.requires_grad_(True), b.requires_grad_(True)) for W, b in
              mlp_init(g, [NF + 2, 16, 1], zero_last=True)]
        phys["g"] = gp

    net_params = [p for pair in netp for p in pair]
    g_params = [p for pair in gp for p in pair] if gp else []
    opt = torch.optim.Adam([
        {"params": net_params, "lr": 2e-3, "name": "net"},
        {"params": list(phys.values()) if not gp else
         [v for k, v in phys.items() if k != "g"] + g_params, "lr": 1e-2, "name": "phys"},
    ])
    # cosine decay to alpha*lr0 (alpha=0.05), one lambda per group
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, [lambda x: 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * x / n_steps))] * 2)

    def U(netp, oh):
        o = mlp(netp, oh)
        return torch.stack([mu_e + sd_e * o[:, 0], mu_c + sd_c * o[:, 1]], dim=1)

    se, sc = 0.03, 0.002  # residual scales [K/s]

    def loss_fn(w_phys):
        # data term
        pred = U(netp, oh_net)
        l_data = torch.mean(((pred[:, 0] - Ted_t) / sd_e) ** 2) + torch.mean(((pred[:, 1] - Tcd_t) / sd_c) ** 2)
        # physics residual with dU/dt via autodiff (batched: grad of column sums)
        t_req = tc_t.clone().requires_grad_(True)
        o = mlp(netp, torch.cat([t_req[:, None] / T0, tc_sidoh], dim=1))
        y0c = mu_e + sd_e * o[:, 0]
        y1c = mu_c + sd_c * o[:, 1]
        dy0 = torch.autograd.grad(y0c.sum(), t_req, create_graph=True)[0]
        dy1 = torch.autograd.grad(y1c.sum(), t_req, create_graph=True)[0]
        dTe, dTc = rhs(phys, tc_fl, y0c, y1c, tc_tv, state_dependent, kappa_te)
        R0 = (dy0 - dTe) / se
        R1 = (dy1 - dTc) / sc
        l_phys = torch.mean(R0 ** 2) + torch.mean(R1 ** 2)
        return l_data + w_phys * lam_phys * l_phys, l_data, l_phys

    hist = []
    n0, n1 = int(0.25 * n_steps), max(1, int(0.25 * n_steps))   # data-only stage, then linear ramp
    for i in range(n_steps):
        w = float(np.clip((i - n0) / n1, 0.0, 1.0))
        opt.zero_grad()
        L, ld, lp = loss_fn(w)
        L.backward()
        opt.step(); sched.step()
        if i % max(1, n_steps // 12) == 0 or i == n_steps - 1:
            hist.append((i, float(ld.detach()), float(lp.detach())))
            if verbose:
                print(f"  it {i:5d}  data {float(ld):.5f}  phys {float(lp):.4f}  "
                      f"a={np.round(np.exp(phys['loga'].detach().numpy()), 4)}  "
                      f"K={np.round(np.exp(phys['logK'].detach().numpy()), 5)}", flush=True)
    phys_out = {k: (mlp_np(v) if k == "g" else v.detach().numpy().copy()) for k, v in phys.items()}
    return dict(phys=phys_out, norm=norm, hist=hist, state_dependent=state_dependent)


def phys_summary(phys):
    a = np.exp(np.array(phys["loga"])); K = np.exp(np.array(phys["logK"]))
    d = np.exp(np.array(phys["logd"]))
    out = dict(a=a, K=K, r=float(np.exp(phys["logr"])), d=float(d) if d.ndim == 0 else d,
               Tcool=float(phys["Tcool"]), Rstar=a / K)   # R* at Te=50 (kappa(50)=k0)
    if "beta" in phys:
        out["beta"] = np.array(phys["beta"])              # per-fluid 1/degC
    return out
