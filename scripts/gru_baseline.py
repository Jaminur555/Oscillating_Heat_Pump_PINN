"""Tier 2 (7): GRU black-box baseline, LORO evaluation. Results -> runs/outer/loro_torch.pkl
(keys 'GRU|seedN' per fold, same structure as PINN entries). Log: runs/outer/gru_baseline.log."""
import os, pickle, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from ohp.data import load_all, FLUIDS, RUNS
from ohp.gru import fit_gru
from ohp.metrics import metrics_raw as metrics

OUT = "runs/outer/loro_torch.pkl"
SEEDS = [0, 1, 2]
D = load_all()
res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}

def save(): pickle.dump(res, open(OUT, "wb"))

def log(msg): print(msg, flush=True)

folds = [1, 2, 3, None]            # None = fit on all runs (final model)
for ho in folds:
    tr = [((r, f), D[(r, f)]) for r in RUNS for f in FLUIDS if r != ho]
    te = [((r, f), D[(r, f)]) for r in RUNS for f in FLUIDS if r == ho]
    R = res.setdefault(ho, {})
    for seed in SEEDS:
        name = f"GRU|seed{seed}"
        if name in R:
            log(f"[fold {ho}] {name} cached"); continue
        t0 = time.time()
        pred = fit_gru(tr, seed=seed, log=(lambda m: log(m)) if seed == 0 else None)
        out = dict(pred={}, met={})
        for k, d in te:
            te_p, tc_p = pred(d, k[1])
            out["pred"][k] = (te_p, tc_p); out["met"][k] = metrics(d, te_p, tc_p)
        R[name] = out; save()
        log(f"[fold {ho}] {name} done in {time.time()-t0:.0f}s")

print("\n=== GRU LORO summary (mean over seeds, RMSE_Te per held-out run) ===", flush=True)
for ho in [1, 2, 3]:
    for fl in FLUIDS:
        vals = [res[ho][f"GRU|seed{s}"]["met"][(ho, fl)]["rmse_Te"] for s in SEEDS if f"GRU|seed{s}" in res[ho]]
        if vals:
            log(f"fold {ho} {fl:5s}: {np.mean(vals):6.2f} +- {np.std(vals):.2f} K  (n={len(vals)})")
log("ALL DONE")
