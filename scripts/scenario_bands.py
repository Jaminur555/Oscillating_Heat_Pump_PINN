"""Review round 2, item 4: how much do the scenario metrics depend on the sloppy ABSOLUTE rates?

The identifiability analysis (fig5) shows a joint rescale of (a_f, kappa_f) by ~x2 barely moves
held-out RMSE -- but scenario timescale metrics (lag, settle, ripple) are set by the absolute
rates, so they carry a x/div-2 band even though R* and the steady-state behaviour are fixed.
This script propagates that band, plus the jackknife leave-one-series-out parameter spread,
through the scenario case study:

  variants = {rate x0.5, x1, x2 on the final fit}  U  {9 leave-one-series-out refits}
             (from runs/outer/jackknife_fits.csv, written by bootstrap_ci.py)

and reports, per case and metric, the fluid ordering stability across variants.
Out: runs/outer/scenario_bands.csv + .log
Run from repo root:  KMP_DUPLICATE_LIB_OK=TRUE python scripts/scenario_bands.py
"""
import os, pickle, sys, tempfile, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from ohp.ode_twin import export_onnx, ODETwin, FLUIDS
from scenario_case import ramp_hold, cyclic, slug, run_case

RES = "runs/outer"
phys_full = pickle.load(open(f"{RES}/loro_torch.pkl", "rb"))[None]["ODE-shoot"]["phys"]

variants = {}
for s in (0.5, 1.0, 2.0):
    p = dict(phys_full)
    p["loga"] = np.asarray(p["loga"], float) + np.log(s)
    p["logK"] = np.asarray(p["logK"], float) + np.log(s)
    variants[f"rate x{s:g}"] = p
J = pd.read_csv(f"{RES}/jackknife_fits.csv")
for i, row in J.iterrows():
    if i == 0:
        continue                                        # row 0 = full fit, covered by "rate x1"
    variants[f"loo-{i}"] = dict(loga=row[["loga_" + f for f in FLUIDS]].values.astype(float),
                                logK=row[["logK_" + f for f in FLUIDS]].values.astype(float),
                                logr=float(row["logr"]), logd=float(row["logd"]), Tcool=float(row["Tcool"]))

CASES = []
for Tset in (60.0, 75.0, 90.0):
    for tau in (150.0, 300.0, 600.0):
        CASES.append((f"A_ramp{Tset:.0f}_tau{tau:.0f}",) + ramp_hold(Tset, tau) + (Tset, int(1800)))
    CASES.append((f"B_cyc{Tset:.0f}",) + cyclic(Tset) + (Tset, int(3600)))
    CASES.append((f"C_slug{Tset:.0f}",) + slug(Tset) + (Tset, int(1800)))

rows = []
with tempfile.TemporaryDirectory() as tmp:
    twins = {}
    for vn, p in variants.items():
        path = os.path.join(tmp, f"twin_{vn.replace(' ', '_').replace('/', '-')}.onnx")
        export_onnx(p, False, path)
        twins[vn] = ODETwin(path)
        print("exported", vn, flush=True)
    for vn, twin in twins.items():
        for name, t, Tv, Tset, i0 in CASES:
            for fl in FLUIDS:
                m = run_case(twin, fl, t, Tv, Tset, (i0, len(t)), name.startswith("A_"))
                rows.append(dict(variant=vn, case=name, fluid=fl, Tset=Tset, **m))

B = pd.DataFrame(rows)
B.to_csv(f"{RES}/scenario_bands.csv", index=False)

print("\n=== headline metrics (final fit, rate-rescale band, jackknife LOO band) ===")
LOW = ["dev_mae", "ripple", "lag_K", "settle_s", "dev_max"]
base = B[B.variant == "rate x1"]
rescale = B[B.variant.str.startswith("rate")]
loo = B[B.variant.str.startswith("loo-")]
for case in sorted(base.case.unique()):
    for m in LOW:
        line = f"{case:18s} {m:9s}"
        for fl in FLUIDS:
            b = base[(base.case == case) & (base.fluid == fl)][m].iloc[0]
            rs = rescale[(rescale.case == case) & (rescale.fluid == fl)][m]
            lo = loo[(loo.case == case) & (loo.fluid == fl)][m]
            line += f"  {fl[:1]}: {b:6.2f} [{rs.min():6.2f},{rs.max():6.2f}] [{lo.min():6.2f},{lo.max():6.2f}]"
        print(line)

print("\n=== ordering stability: best fluid per (case, metric) across variants ===")
for m in LOW:
    piv = B.pivot_table(index=["variant", "case"], columns="fluid", values=m)
    best = piv.idxmin(axis=1)
    print(f"{m:9s}:", best.value_counts(normalize=True).round(2).to_dict())
print("DONE")
