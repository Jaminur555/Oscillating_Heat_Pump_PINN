"""LORO evaluation, PyTorch port. Results -> runs/outer/loro_torch.pkl (resumable).

Only runs framework-affected methods (ODE-shoot integration + PINN); MLP/GP/SINDy
baselines are framework-free and already recorded in loro.pkl / loro_metrics_long.csv.
"""
import os, pickle, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from ohp.data import load_all, FLUIDS, RUNS
from ohp.pinn import train_pinn, simulate_series, phys_summary
from ohp.baselines import fit_ode_shoot
from ohp.metrics import metrics_raw as metrics
from ohp.config import RUNS_DIR as RES

OUT = RES + "/loro_torch.pkl"
SEEDS = list(range(10))
N_STEPS = 6000
D = load_all()
res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}

def save(): pickle.dump(res, open(OUT, "wb"))

folds = [1, 2, 3, None]            # None = fit on all runs (final params)
for ho in folds:
    tr = [((r, f), D[(r, f)]) for r in RUNS for f in FLUIDS if r != ho]
    te = [((r, f), D[(r, f)]) for r in RUNS for f in FLUIDS if r == ho]
    R = res.setdefault(ho, {})
    def run(name, fn):
        if name in R: return
        t0 = time.time(); R[name] = fn(); save(); print(f"[fold {ho}] {name} done in {time.time()-t0:.0f}s", flush=True)

    def do_shoot():
        phys, s = fit_ode_shoot(tr)
        out = dict(phys=phys, pred={}, met={})
        for k, d in te:
            p = simulate_series(phys, d, k[1], False); out["pred"][k] = p; out["met"][k] = metrics(d, *p)
        return out
    run("ODE-shoot", do_shoot)

    def do_pinn(sd_flag, seed, d_shared, kappa_te=False):
        def f():
            M = train_pinn(tr, state_dependent=sd_flag, n_steps=N_STEPS, seed=seed, verbose=False,
                           d_shared=d_shared, kappa_te=kappa_te)
            out = dict(phys=M["phys"], summary=phys_summary(M["phys"]), hist=M["hist"], pred={}, met={})
            for k, d in te:
                p = simulate_series(M["phys"], d, k[1], sd_flag, kappa_te=kappa_te)
                out["pred"][k] = p; out["met"][k] = metrics(d, *p)
            return out
        return f
    for sd_flag, nm in [(False, "PINN-const"), (True, "PINN-state")]:
        for seed in SEEDS:
            run(f"{nm}|seed{seed}", do_pinn(sd_flag, seed, True))
    for seed in [0, 1, 2]:                                # Tier 2 (6): kappa(Te)=k0*exp(beta*(Te-50))
        run(f"PINN-kTe|seed{seed}", do_pinn(False, seed, True, kappa_te=True))
    if ho is not None:                                   # ablation: per-fluid condenser loss rate d
        for seed in [0, 1, 2]:
            run(f"PINN-const-df|seed{seed}", do_pinn(False, seed, False))
print("ALL DONE", flush=True)
