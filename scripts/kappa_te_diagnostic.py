"""Tier 2 diagnostics: (A) per-run kappa(Te) vs const-kappa fits; (B) equal-coupling plateau power.

SUPERSEDED (review round 2): section (A) STALLED -- beta initialised at exactly 0 with a
relative finite-difference step (~0 there) and too few iterations for 6 parameters, so it
reported "no identifiable T-dependence" as an optimiser artefact.  The correct profile
(fixed beta grid, other params refit per point) is scripts/kappa_te_profile.py; beta IS
identifiable within runs.  Section (B) (equal-coupling plateau check) is unaffected.
Kept for the record; do not cite (A) without kappa_te_profile.log.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import pandas as pd
from ohp.data import load_all, FLUIDS, RUNS, RAW_DIR, _load_sheet

D = load_all()

def roll(d, a, kap_fn, r, dd, Tcool):
    Tv = CubicSpline(d["t"], d["Tv"])
    def f(t, y):
        q = kap_fn(y[0]) * (y[0] - y[1])
        return [a * (Tv(t) - y[0]) - q, r * q - dd * (y[1] - Tcool)]
    return solve_ivp(f, [d["t"][0], d["t"][-1]], [d["Te"][0], d["Tc"][0]], t_eval=d["t"],
                     rtol=1e-6, atol=1e-8, method="LSODA").y

print("=== A) per-run fits: const kappa vs kappa(Te)=k0*exp(beta*(Te-50)) ===")
print("run fluid | k_const(1e-3) | k0(1e-3)  beta(1e-3) | RMSE const -> kTe   meanTe")
for r in RUNS:
    for fl in FLUIDS:
        d = D[(r, fl)]
        def res_const(p):
            P = np.exp(p)
            y = roll(d, P[0], lambda Te: P[1], P[2], P[3], P[4])
            return np.r_[y[0] - d["Te"], 4 * (y[1] - d["Tc"])]
        def res_kte(p):
            P = np.exp(p[:5]); b = p[5]
            y = roll(d, P[0], lambda Te: P[1] * np.exp(b * (Te - 50.0)), P[2], P[3], P[4])
            return np.r_[y[0] - d["Te"], 4 * (y[1] - d["Tc"])]
        p0 = np.log([0.015, 0.0016, 0.03, 0.003, 18.8])
        sc = least_squares(res_const, p0, diff_step=1e-3, max_nfev=60)
        sk = least_squares(res_kte, np.r_[p0, 0.0], diff_step=1e-4, max_nfev=120)
        rc = np.sqrt(np.mean(sc.fun[:len(d["t"])] ** 2))
        rk = np.sqrt(np.mean(sk.fun[:len(d["t"])] ** 2))
        print(f"{r}   {fl:5s} | {np.exp(sc.x[1])*1e3:8.4f}     | {np.exp(sk.x[1])*1e3:7.4f} {sk.x[5]*1e3:+8.3f} "
              f"| {rc:.3f} -> {rk:.3f}   {d['Te'].mean():.1f}")

print("\n=== B) equal-coupling: plateau vessel dT (inner-outer) vs coupling dT ===")
print("run fluid | <Ti-To> K (sd) | <Tv-Te> K | <Tv-Tc> K   (plateau = last 25%)")
for r in RUNS:
    for fl in FLUIDS:
        df = pd.read_excel(f"{RAW_DIR}/HCOHP Primary Data - Run {r}.xlsx", sheet_name=f"{fl} R{r}", header=5)
        ti = df[[c for c in df.columns if "Inner" in str(c)]].mean(axis=1)
        to = df[[c for c in df.columns if "Outer" in str(c)]].mean(axis=1)
        d = D[(r, fl)]
        n = len(d["t"]); k0, k1 = int(0.75 * n), n
        dT = (ti - to).values
        dT = dT[int(0.75 * len(dT)):]
        print(f"{r}   {fl:5s} | {dT.mean():7.3f} ({dT.std():.3f}) | {(d['Tv']-d['Te'])[k0:].mean():7.3f}  | {(d['Tv']-d['Tc'])[k0:].mean():7.3f}")
