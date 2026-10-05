"""Re-fit the MLP/GP baselines WITH initial-condition features (fair-baseline fix) and recompute
ALL stored LORO metrics against the RAW thermocouple data. Framework-free: cached ODE/PINN
predictions are unchanged by both fixes, so only sklearn is re-run here.
Usage: python rerun_baselines.py   (writes results/loro.pkl, loro_metrics_long.csv, loro_summary.csv,
figures/loro_rmse.png, figures/loro_predictions.png)
"""
import pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ohp.data import load_all, FLUIDS
from ohp.baselines import fit_mlp, fit_gp
from ohp.metrics import metrics_raw
from ohp.config import RUNS_DIR as RES, FIG_DIR as FIG

RUNS = [1, 2, 3]
D = load_all()
res = pickle.load(open(RES + "/loro.pkl", "rb"))

# 1. drop and refit the black-box baselines (now with IC features)
for ho in RUNS:
    R = res[ho]
    tr = [((r, f), D[(r, f)]) for r in RUNS for f in FLUIDS if r != ho]
    te = [((r, f), D[(r, f)]) for r in RUNS for f in FLUIDS if r == ho]
    for name, fitter in [("MLP", fit_mlp), ("GP", fit_gp)]:
        R.pop(name, None)
        pred_fn = fitter(tr); out = dict(pred={}, met={})
        for k, d in te:
            p = pred_fn(d, k[1]); out["pred"][k] = p; out["met"][k] = metrics_raw(d, *p)
        R[name] = out
        print(f"[fold {ho}] {name} refit with IC features", flush=True)

# 2. recompute metrics for every cached entry against the raw data
for ho in RUNS + [None]:
    for name, v in res[ho].items():
        for k, p in v["pred"].items():
            v["met"][k] = metrics_raw(D[k], np.asarray(p)[0], np.asarray(p)[1])
pickle.dump(res, open(RES + "/loro.pkl", "wb"))

# 3. refresh summary CSVs + the two LORO figures
rows = [dict(fold=ho, method=name.split("|")[0], seed=name, fluid=k[1], **mt)
        for ho in RUNS for name, v in res[ho].items() for k, mt in v["met"].items()]
df = pd.DataFrame(rows); df.to_csv(RES + "/loro_metrics_long.csv", index=False)
g = df.groupby(["fold", "method", "seed"])[["rmse_Te", "rmse_Tc", "end_err_Te", "max_abs_Te"]].mean().reset_index()
tab = g.groupby(["fold", "method"])[["rmse_Te", "rmse_Tc", "end_err_Te", "max_abs_Te"]].agg(["mean", "std"]).round(2)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("\nTe/Tc RMSE vs RAW data (mean over fluids; +-sd over 2 seeds where present):")
print(tab.to_string())
t = df.groupby(["fold", "method", "seed"])[["rmse_Te", "rmse_Tc"]].mean().reset_index()
t2 = t.groupby(["fold", "method"])[["rmse_Te", "rmse_Tc"]].agg(["mean", "std"])
t2.round(2).to_csv(RES + "/loro_summary.csv")

COL = {"ODE-shoot": "C1", "PINN-const": "C3", "PINN-state": "C4", "GP": "C2", "MLP": "C0"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k, (v, ttl) in enumerate([("rmse_Te", "Evaporator Te RMSE vs raw data [K]"), ("rmse_Tc", "Condenser Tc RMSE vs raw data [K]")]):
    for mi, m in enumerate(["MLP", "GP", "ODE-shoot", "PINN-const", "PINN-state"]):
        ys = [t2.loc[(f, m), (v, "mean")] for f in RUNS]; es = [np.nan_to_num(t2.loc[(f, m), (v, "std")]) for f in RUNS]
        ax[k].bar(np.arange(3) + mi * 0.16 - 0.32, ys, 0.16, yerr=es, color=COL[m], label=m)
    ax[k].set_xticks(range(3)); ax[k].set_xticklabels(["hold out Run 1", "hold out Run 2", "hold out Run 3\n(extrapolation)"])
    ax[k].set_yscale("log"); ax[k].set_title(ttl); ax[k].grid(alpha=.3, axis="y")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG + "/loro_rmse.png", dpi=110); plt.close()

fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for i, ho in enumerate(RUNS):
    for j, fl in enumerate(FLUIDS):
        d = D[(ho, fl)]; a = ax[i, j]
        a.plot(d["raw"]["t"], d["raw"]["Te"], color="0.75", lw=0.6)
        a.plot(d["t"], d["Te"], "k", lw=2.0, label="measured Te (smoothed; grey: raw)")
        for m in ["MLP", "GP", "ODE-shoot", "PINN-const", "PINN-state"]:
            nm = m if m in res[ho] else m + "|seed0"
            a.plot(d["t"], np.asarray(res[ho][nm]["pred"][(ho, fl)])[0], color=COL[m], lw=1.3, label=m)
        a.set_title(f"held-out Run {ho} - {fl}" + ("  (EXTRAPOLATION)" if ho == 3 else ""), fontsize=10)
        if i == 2: a.set_xlabel("time [s]")
        if j == 0: a.set_ylabel("Te [°C]")
        if i == 0 and j == 0: a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "/loro_predictions.png", dpi=110); plt.close()
print("\nfigures/loro_rmse.png, figures/loro_predictions.png and summary CSVs refreshed")
