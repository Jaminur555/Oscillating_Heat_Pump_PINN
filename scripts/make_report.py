import pickle, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from scipy.interpolate import CubicSpline
from ohp.data import load_all, FLUIDS, RUNS
from ohp.config import RUNS_DIR as RES, FIG_DIR as FIG
from ohp.pinn import simulate_series, rhs
import ohp.ode_twin as TW

D = load_all()
# framework-free baselines (MLP/GP/SINDy) live in loro.pkl; the live torch pipeline
# (ODE-shoot / PINN / GRU) in loro_torch.pkl overrides the retired JAX entries.
res = pickle.load(open(RES + "/loro.pkl", "rb"))
for ho, R in pickle.load(open(RES + "/loro_torch.pkl", "rb")).items():
    res.setdefault(ho, {}).update(R)
COL = {"ODE-shoot": "C1", "PINN-const": "C3", "PINN-state": "C4", "GP": "C2", "MLP": "C0"}
out = {}

# ---------------- 1. LORO figures -------------------------------------------------------------
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for i, ho in enumerate([1, 2, 3]):
    for j, fl in enumerate(FLUIDS):
        d = D[(ho, fl)]; a = ax[i, j]
        a.plot(d["t"], d["Te"], "k", lw=2.5, label="measured Te")
        for m in ["MLP", "GP", "ODE-shoot", "PINN-const", "PINN-state"]:
            nm = m if m in res[ho] else m + "|seed0"
            a.plot(d["t"], np.asarray(res[ho][nm]["pred"][(ho, fl)])[0], color=COL[m], lw=1.3, label=m)
        a.set_title(f"held-out Run {ho} - {fl}" + ("  (EXTRAPOLATION)" if ho == 3 else ""), fontsize=10)
        if i == 2: a.set_xlabel("time [s]")
        if j == 0: a.set_ylabel("Te [°C]")
        if i == 0 and j == 0: a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG + "/loro_predictions.png", dpi=110); plt.close()

df = pd.read_csv(RES + "/loro_metrics_long.csv")
g = df.groupby(["fold", "method", "seed"])[["rmse_Te", "rmse_Tc"]].mean().reset_index()
t = g.groupby(["fold", "method"])[["rmse_Te", "rmse_Tc"]].agg(["mean", "std"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k, (v, ttl) in enumerate([("rmse_Te", "Evaporator Te RMSE [K]"), ("rmse_Tc", "Condenser Tc RMSE [K]")]):
    for mi, m in enumerate(["MLP", "GP", "ODE-shoot", "PINN-const", "PINN-state"]):
        ys = [t.loc[(f, m), (v, "mean")] for f in [1, 2, 3]]; es = [np.nan_to_num(t.loc[(f, m), (v, "std")]) for f in [1, 2, 3]]
        ax[k].bar(np.arange(3) + mi * 0.16 - 0.32, ys, 0.16, yerr=es, color=COL[m], label=m)
    ax[k].set_xticks(range(3)); ax[k].set_xticklabels(["hold out Run 1", "hold out Run 2", "hold out Run 3\n(extrapolation)"])
    ax[k].set_yscale("log"); ax[k].set_title(ttl); ax[k].grid(alpha=.3, axis="y")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG + "/loro_rmse.png", dpi=110); plt.close()
tab = t.round(2); tab.to_csv(RES + "/loro_summary.csv"); out["loro_table"] = tab.reset_index().to_dict("records") if False else None

# ---------------- 2. R* (normalised pipe resistance) ------------------------------------------
rows = []
def Rs(phys): return np.exp(np.asarray(phys["loga"]) - np.asarray(phys["logK"]))
for ho in [1, 2, 3, None]:
    for nm, v in res[ho].items():
        if nm.startswith("PINN-const") or nm == "ODE-shoot":
            for f, r in zip(FLUIDS, Rs(v["phys"])): rows.append(dict(fold=str(ho), model=nm.split("|")[0], fluid=f, Rstar=float(r)))
R = pd.DataFrame(rows); R.to_csv(RES + "/Rstar_all_fits.csv", index=False)
Rsum = R.groupby(["model", "fluid"]).Rstar.agg(["mean", "std", "min", "max"]).round(2); print(Rsum)
Rsum.to_csv(RES + "/Rstar_summary.csv")
fig, ax = plt.subplots(figsize=(6, 4))
for k, fl in enumerate(FLUIDS):
    v = R[R.fluid == fl]
    ax.scatter(np.full(len(v), k) + np.random.RandomState(0).uniform(-.12, .12, len(v)), v.Rstar, c=[("C1" if m == "ODE-shoot" else "C3") for m in v.model], s=25)
ax.set_xticks(range(3)); ax.set_xticklabels(FLUIDS); ax.set_ylabel("R* = a/κ  (R_pipe / R_in)"); ax.grid(alpha=.3)
ax.set_title("Normalised pipe resistance, every fit (orange: ODE, red: PINN)"); plt.tight_layout(); plt.savefig(FIG + "/Rstar.png", dpi=110); plt.close()

# ---------------- 3. identifiability diagnostic -----------------------------------------------
ph = res[None]["ODE-shoot"]["phys"]; scales = np.exp(np.linspace(np.log(0.25), np.log(4), 15)); cur = []
for s in scales:
    p = dict(ph); p["loga"] = np.asarray(ph["loga"]) + np.log(s); p["logK"] = np.asarray(ph["logK"]) + np.log(s)
    e = [np.sqrt(np.mean((simulate_series(p, D[(r, f)], f, False)[0] - D[(r, f)]["Te"]) ** 2)) for r in RUNS for f in FLUIDS]
    cur.append(np.mean(e))
fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(scales, cur, "o-"); ax.set_xscale("log"); ax.grid(alpha=.3)
ax.set_xlabel("common scale factor on (a, K)  [R* held fixed]"); ax.set_ylabel("mean Te RMSE over all 9 series [K]")
ax.set_title("Identifiability: rates are sloppy, ratio is not"); plt.tight_layout(); plt.savefig(FIG + "/identifiability.png", dpi=110); plt.close()
out["ident"] = dict(scales=scales.tolist(), rmse=[float(c) for c in cur])
print("identifiability: RMSE at scale 0.5/1/2 =", np.round(np.interp([0.5, 1, 2], scales, cur), 2))

# ---------------- 4. digital twin: ONNX export, equivalence check, replay, sweep ---------------
def phys_np(v): return dict(loga=np.asarray(v["phys"]["loga"]), logK=np.asarray(v["phys"]["logK"]), logr=float(v["phys"]["logr"]),
                            logd=float(v["phys"]["logd"]), Tcool=float(v["phys"]["Tcool"]))
paths = {}
for ho in [1, 2, 3, None]:
    p = RES + f"/ode_twin_fold{ho}.onnx"; TW.export_onnx(phys_np(res[ho]["PINN-const|seed0"]), False, p); paths[ho] = p
# equivalence ONNX vs torch rhs
tw = TW.ODETwin(paths[None]); P = phys_np(res[None]["PINN-const|seed0"]); rng = np.random.RandomState(0); worst = 0
for _ in range(200):
    fl = FLUIDS[rng.randint(3)]; Te, Tc, Tv = rng.uniform(15, 90), rng.uniform(15, 25), rng.uniform(15, 95)
    a = tw.f(Te, Tc, Tv, fl)[:2]
    Pt = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in P.items()}
    b = rhs(Pt, torch.eye(3)[FLUIDS.index(fl)], float(Te), float(Tc), float(Tv), False)
    worst = max(worst, abs(a[0] - float(b[0])), abs(a[1] - float(b[1])))
print("max |ONNX - torch| derivative difference: %.2e K/s" % worst); out["onnx_max_diff"] = float(worst)

# streaming replay of the held-out Run 3 with the fold-3 twin (never trained on Run 3)
tw3 = TW.ODETwin(paths[3]); fig, ax = plt.subplots(1, 3, figsize=(15, 4)); rep = {}
for j, fl in enumerate(FLUIDS):
    d = D[(3, fl)]; tt = np.arange(d["t"][0], d["t"][-1] + 1e-9, 1.0)
    Tv1 = CubicSpline(d["t"], d["Tv"])(tt); Te, Tc, Rs_ = tw3.replay(fl, Tv1, [d["Te"][0], d["Tc"][0]])
    pe = np.interp(d["t"], tt, Te); rep[fl] = float(np.sqrt(np.mean((pe - d["Te"]) ** 2)))
    ax[j].plot(d["t"], d["Te"], "k", lw=2, label="measured Te"); ax[j].plot(tt, Te, "C3", label="twin (ONNX, streaming)")
    ax[j].plot(tt, Tv1, "k--", lw=1, label="Tv input"); ax[j].set_title(f"Run 3 {fl}: RMSE {rep[fl]:.2f} K (out-of-sample)"); ax[j].set_xlabel("time [s]")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG + "/twin_replay_run3.png", dpi=110); plt.close()
out["twin_run3_rmse"] = rep; print("twin replay Run 3 RMSE:", rep)

# fluid sweep with all 4 fitted parameter sets (spread = parameter uncertainty)
rows = []
for ho, p in paths.items():
    sw = TW.sweep(TW.ODETwin(p))
    for (fl, si), m in sw.items(): rows.append(dict(fitset=str(ho), fluid=fl, scen=si, **m))
S = pd.DataFrame(rows); S.to_csv(RES + "/fluid_sweep.csv", index=False)
agg = S.groupby("fluid")[["peakTe", "gap", "Rstar"]].agg(["mean", "std"]).round(2); print(agg)
# ranking consistency: in how many (fitset, scenario) cases is each fluid best / worst on each metric
cons = {}
for met, better in [("peakTe", "min"), ("gap", "max"), ("Rstar", "min")]:
    piv = S.pivot_table(index=["fitset", "scen"], columns="fluid", values=met)
    best = piv.idxmin(axis=1) if better == "min" else piv.idxmax(axis=1)
    cons[met] = best.value_counts(normalize=True).round(2).to_dict()
print("share of cases each fluid is best:", cons); out["best_share"] = cons
json.dump(out, open(RES + "/report_numbers.json", "w"), indent=1, default=str)
