"""Publish-quality figures for the paper (paper/figures/*.pdf + 300 dpi PNG previews).

Style rules (fixed after review): pure white background; NO black curves (measured =
dark gray); in comparison figures ALL curves share one linewidth, distinguished by
colour only; legends ALWAYS outside the axes (below the figure), never inside;
despined axes; Okabe-Ito palette; 8 pt serif. Regenerates from the pickles in
runs/outer, no fitting.  Run from repo root:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/make_paper_figures.py
"""
import os, sys, json, pickle, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

sys.path.insert(0, "scripts")
from ohp.data import load_all, FLUIDS, RUNS
from ohp.ode_twin import ODETwin
from scenario_case import ramp_hold, cyclic, slug

FIG = "paper/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "stix",
                     "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "text.color": "#222222", "axes.labelcolor": "#222222",
                     "xtick.color": "#444444", "ytick.color": "#444444", "axes.edgecolor": "#777777",
                     "axes.grid": False,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white",
                     "legend.frameon": True, "legend.framealpha": 1.0,
                     "legend.edgecolor": "#CCCCCC", "legend.fancybox": False,
                     "lines.linewidth": 1.0, "lines.markersize": 3.5,
                     "savefig.bbox": "tight", "savefig.dpi": 300})
FL = dict(EOHP="#E69F00", MOHP="#009E73", WOHP="#0072B2")        # fluids
ME = dict(meas="#555555", ode="#0072B2", pinn="#D55E00", gru="#009E73",
          mlp="#CC79A7", gp="#56B4E9", sindy="#E69F00")           # methods
LW = 1.0                                                          # one width for all curves
D = load_all()
lj = pickle.load(open("runs/outer/loro.pkl", "rb"))
rt = pickle.load(open("runs/outer/loro_torch.pkl", "rb"))
RES = {ho: {**lj.get(ho, {}), **rt.get(ho, {})} for ho in [1, 2, 3, None]}


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf"); fig.savefig(f"{FIG}/{name}.png", dpi=300); plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------- fig1b: raw data overview (standalone)
fig, a = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
CHS = [("Tv", r"vessel $T_v$", "-", "#888888"), ("Ta", r"adiabatic $T_a$", "-.", "#AAAAAA"),
       ("Tc", r"condenser $T_c$", ":", "#666666")]
d0 = D[(2, FLUIDS[0])]
for ch, lb, ls, c in CHS:                      # driver channels: shared bath, plot once
    a.plot(d0["t"] / 60, d0[ch], ls=ls, color=c, lw=LW, label=lb)
for fl in FLUIDS:                              # evaporator: one curve per fluid
    d = D[(2, fl)]
    a.plot(d["t"] / 60, d["Te"], color=FL[fl], lw=LW, label=r"$T_e$, " + fl)
a.set_xlabel("time [min]"); a.set_ylabel(r"temperature [$^\circ$C]")
fig.legend(loc="outside lower center", ncol=3, fontsize=6.5, frameon=False)
save(fig, "fig1_data")

# ------------------------------------------------------- fig2: LORO predictions
fig, axs = plt.subplots(3, 3, figsize=(7.0, 5.8), sharex=True, constrained_layout=True)
for i, ho in enumerate(RUNS):
    for j, fl in enumerate(FLUIDS):
        d = D[(ho, fl)]; a = axs[i, j]
        a.plot(d["t"] / 60, d["Te"], color=ME["meas"], lw=LW, label="measured")
        a.plot(d["t"] / 60, RES[ho]["ODE-shoot"]["pred"][(ho, fl)][0], color=ME["ode"], lw=LW, label="ODE-shoot")
        pinn = np.mean([RES[ho][f"PINN-const|seed{s}"]["pred"][(ho, fl)][0] for s in range(10)], 0)
        a.plot(d["t"] / 60, pinn, color=ME["pinn"], lw=LW, label="PINN (seed mean)")
        gru = np.mean([RES[ho][f"GRU|seed{s}"]["pred"][(ho, fl)][0] for s in range(3)], 0)
        a.plot(d["t"] / 60, gru, color=ME["gru"], lw=LW, label="GRU (seed mean)")
        if i == 0: a.set_title(fl)
        if j == 0: a.set_ylabel(r"$T_e$ [$^\circ$C]")
        if i == 2: a.set_xlabel("time [min]")
        if ho == 3 and j == 2:
            a.text(0.04, 0.9, "extrapolation fold", transform=a.transAxes, fontsize=7, style="italic", va="top")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=7, frameon=False)
save(fig, "fig2_loro")

# ------------------------------------------------------- fig3: RMSE summary (rebuilt)
def fold_means(pref, res, seeded, n=10):
    out = []
    for ho in RUNS:
        ks = [f"{pref}|seed{s}" for s in range(n) if f"{pref}|seed{s}" in res[ho]] if seeded else [pref]
        v = [np.mean([res[ho][k]["met"][(ho, fl)]["rmse_Te"] for fl in FLUIDS]) for k in ks]
        out.append((np.mean(v), np.std(v)))
    return np.array(out)
MCOL = {"MLP": ME["mlp"], "GP": ME["gp"], "SINDy": ME["sindy"], "GRU": ME["gru"],
        "ODE-shoot": ME["ode"], "PINN-const": ME["pinn"], "PINN-state": "#8C5100",
        "Trivial": "#999999", "Ridge-linear": "#882255", "One-node": "#BCBD22", "Linear-SS": "#6A51A3"}
MARK = {"MLP": "o", "GP": "s", "SINDy": "D", "GRU": "^", "ODE-shoot": "v", "PINN-const": "P", "PINN-state": "X",
        "Trivial": "|", "Ridge-linear": "1", "One-node": "2", "Linear-SS": "3"}
rb = pickle.load(open("runs/outer/review_baselines.pkl", "rb"))
meths = [("Trivial", rb, 0), ("MLP", lj, 0), ("GP", lj, 0), ("SINDy", lj, 0), ("Ridge-linear", rb, 0),
         ("Linear-SS", rb, 0), ("GRU", rt, 3), ("One-node", rb, 0), ("ODE-shoot", rt, 0),
         ("PINN-const", rt, 10), ("PINN-state", rt, 10)]
fig, a = plt.subplots(figsize=(4.3, 3.0), constrained_layout=True)
a.axhline(1.0, color="#999999", lw=0.6, ls="--")
for k, (nm, res, ns) in enumerate(meths):
    fm = fold_means(nm, res, ns > 1, ns)
    x = np.arange(3) + (k - (len(meths) - 1) / 2) * 0.1
    a.scatter(x, fm[:, 0], color=MCOL[nm], marker=MARK[nm], s=14, alpha=0.9,
              label=nm + (" (seeds)" if ns > 1 else ""))
    if fm[:, 1].max() > 0:                                       # sd whiskers only where seeds exist
        a.errorbar(x, fm[:, 0], yerr=fm[:, 1], color=MCOL[nm], lw=0.7, capsize=1.5,
                   alpha=0.9, fmt="none")
a.set_yscale("log"); a.set_xticks(range(3)); a.set_xticklabels(["hold out R1", "hold out R2", "hold out R3 (extrap.)"])
a.set_xlim(-0.6, 2.6); a.set_ylabel(r"held-out $T_e$ RMSE [K]")
a.text(2.55, 1.07, "1 K", fontsize=6, color="#888888", ha="right")
fig.legend(loc="outside lower center", ncol=4, fontsize=6.5, frameon=False)
save(fig, "fig3_rmse")

# ------------------------------------------------------- fig4: R* strip
R = pd.read_csv("runs/outer/Rstar_all_fits.csv")
fig, a = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
for k, fl in enumerate(FLUIDS):
    v = R[R.fluid == fl]
    x = k + np.random.RandomState(0).uniform(-0.13, 0.13, len(v))
    a.scatter(x, v.Rstar, c=[ME["ode"] if m == "ODE-shoot" else ME["pinn"] for m in v.model], s=12, alpha=0.8)
    a.scatter([k], [v.Rstar.mean()], marker="_", s=250, color="#555555", lw=1.2)
a.set_xticks(range(3)); a.set_xticklabels(FLUIDS); a.set_ylabel(r"$R^* = a_f/\kappa_f$")
a.scatter([], [], c=ME["ode"], s=12, label="ODE-shoot"); a.scatter([], [], c=ME["pinn"], s=12, label="PINN-const")
fig.legend(loc="outside lower center", ncol=2, fontsize=6.5, frameon=False)
save(fig, "fig4_rstar")

# ------------------------------------------------------- fig5: identifiability
ident = json.load(open("runs/outer/report_numbers.json"))["ident"]
fig, a = plt.subplots(figsize=(3.2, 2.5), constrained_layout=True)
a.plot(ident["scales"], ident["rmse"], "o-", ms=3, color=ME["ode"])
a.axvline(1.0, color="#999999", lw=0.6, ls="--")
a.set_xscale("log")
a.set_xlabel("common scale on $(a_f,\\ \\kappa_f)$  [$R^*$ fixed]"); a.set_ylabel("mean $T_e$ RMSE [K]")
save(fig, "fig5_ident")

# ------------------------------------------------------- fig6: twin replay run 3
tw3 = ODETwin("runs/outer/ode_twin_fold3.onnx")
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
for j, fl in enumerate(FLUIDS):
    d = D[(3, fl)]; a = axs[j]; tt = np.arange(d["t"][0], d["t"][-1] + 1e-9, 1.0)
    Tv1 = CubicSpline(d["t"], d["Tv"])(tt)
    Te, Tc, _ = tw3.replay(fl, Tv1, [d["Te"][0], d["Tc"][0]])
    a.plot(d["raw"]["t"] / 60, d["raw"]["Te"], color=ME["meas"], lw=LW, label="measured (raw)")
    a.plot(tt / 60, Te, color=ME["ode"], lw=LW, label="ONNX twin")
    a.set_title(fl); a.set_xlabel("time [min]")
    if j == 0: a.set_ylabel(r"$T_e$ [$^\circ$C]")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
save(fig, "fig6_replay")

# ------------------------------------------------------- fig7: case study
twN = ODETwin("runs/outer/ode_twin_foldNone.onnx")
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True)
panels = [("a", *ramp_hold(75.0, 300.0), 75.0, "ramp + hold, " + r"$\tau=300$ s"),
          ("b", *cyclic(75.0), 75.0, "thermostat duty (last 2 cycles)"),
          ("c", *slug(75.0), 75.0, "disturbance slug")]
w0 = {"a": 0, "b": int(3600), "c": int(1800)}
for (tag, t, Tv, Tset, ttl), a in zip(panels, axs):
    a.plot(t / 60, Tv, color="#888888", lw=LW, label=r"vessel $T_v$")
    a.axhline(Tset, color="#999999", lw=0.6, ls="--", label="setpoint")
    for fl in FLUIDS:
        Te, _, _ = twN.replay(fl, Tv, [Tv[0], 18.0])
        i0 = w0[tag]
        a.plot(t[i0:] / 60, Te[i0:], color=FL[fl], lw=LW, label=fl)
    a.set_title(ttl, fontsize=7.5); a.set_xlabel("time [min]")
    if tag == "a": a.set_ylabel(r"$T$ [$^\circ$C]")
    if tag == "b": a.set_xlim((t[w0["b"]] / 60, t[-1] / 60))
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=5, fontsize=6.5, frameon=False)
save(fig, "fig7_case")

# ------------------------------------------------------- fig8: adiabatic held-out
fig, axs = plt.subplots(3, 3, figsize=(7.0, 5.4), sharex=True, constrained_layout=True)
for i, ho in enumerate(RUNS):
    for j, fl in enumerate(FLUIDS):
        tr = [D[(r, fl)] for r in RUNS if r != ho]
        X = np.vstack([np.c_[d["Te"], d["Tc"]] for d in tr]); y = np.concatenate([d["Ta"] for d in tr])
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        d = D[(ho, fl)]
        te_p, tc_p = rt[ho]["PINN-const|seed0"]["pred"][(ho, fl)]
        ta_p = w[0] * te_p + w[1] * tc_p
        a = axs[i, j]
        a.plot(d["t"] / 60, d["Ta"], color=ME["meas"], lw=LW, label="measured $T_a$")
        a.plot(d["t"] / 60, ta_p, color=ME["pinn"], lw=LW, label="twin-chain $\\hat T_a$")
        rm = np.sqrt(np.mean((ta_p - d["Ta"]) ** 2))
        a.set_title(f"{fl}  RMSE {rm:.2f} K", fontsize=7)
        if j == 0: a.set_ylabel(r"$T_a$ [$^\circ$C]")
        if i == 2: a.set_xlabel("time [min]")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
save(fig, "fig8_adiabatic")

# ------------------------------------------------------- fig9: ensemble bands
fig, a = plt.subplots(figsize=(3.9, 2.6), constrained_layout=True)
d = D[(3, "EOHP")]
P = np.stack([rt[3][f"PINN-const|seed{s}"]["pred"][(3, "EOHP")][0] for s in range(10)])
mu, sd = P.mean(0), P.std(0)
a.fill_between(d["t"] / 60, mu - 2 * sd, mu + 2 * sd, color=ME["pinn"], alpha=0.3, lw=0, label=r"seed band ($\pm2$ sd)")
a.plot(d["t"] / 60, mu, color=ME["pinn"], lw=LW, label="seed mean")
a.plot(d["raw"]["t"] / 60, d["raw"]["Te"], color=ME["meas"], lw=LW, alpha=0.7, label="measured (raw)")
a.set_xlabel("time [min]"); a.set_ylabel(r"$T_e$ [$^\circ$C]")
fig.legend(loc="outside lower center", ncol=3, fontsize=6.5, frameon=False)
save(fig, "fig9_bands")
print("ALL FIGURES DONE")
