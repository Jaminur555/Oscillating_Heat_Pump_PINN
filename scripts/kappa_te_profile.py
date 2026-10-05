"""Review round 2, item 1: proper profile of kappa(Te) = k0*exp(beta*(Te-50)).

The earlier within-run diagnostic stalled (beta initialised at exactly 0 with a *relative*
finite-difference step ~0 there, and max_nfev too small for 6 parameters), which is what
produced the "beta not identifiable" claim. Here beta is FIXED on a grid and the other five
parameters are refitted by ODE shooting at every grid point -- a profile in beta that cannot
stall, plus a transfer test:

(A) per (run, fluid)  : RMSE_Te(beta) profile, warm-started along the grid; beta_hat and the
                        improvement over beta=0 answer "is beta identifiable within a run?"
(B) per fluid, LORO   : (a, k0, r, d, Tcool) refitted on two runs at each beta; the held-out
                        run is then simulated with the training parameters -- answers "does
                        the within-run beta transfer across heating levels?"

Out: runs/outer/kappa_te_profile.csv + .log
Run from repo root:  KMP_DUPLICATE_LIB_OK=TRUE python scripts/kappa_te_profile.py
"""
import numpy as np, pandas as pd, time
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from ohp.data import load_all, FLUIDS, RUNS

D = load_all()
BETA_GRID = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03, 0.04,
             -0.0025, -0.005, -0.01, -0.015, -0.02]          # warm chain: out from 0
P0 = np.log([0.03, 0.0016, 0.03, 0.003, 18.8])               # a, k0, r, d, Tcool


def roll(d, p, beta):
    a, k0, r, dd, Tcool = np.exp(p)
    Tv = CubicSpline(d["t"], d["Tv"])
    def f(t, y):
        q = k0 * np.exp(beta * (y[0] - 50.0)) * (y[0] - y[1])
        return [a * (Tv(t) - y[0]) - q, r * q - dd * (y[1] - Tcool)]
    s = solve_ivp(f, [d["t"][0], d["t"][-1]], [d["Te"][0], d["Tc"][0]], t_eval=d["t"],
                  rtol=1e-6, atol=1e-8, method="LSODA")
    if not s.success or s.y.shape[1] != len(d["t"]):
        # diverged rollout at this (params, beta): return a large finite curve so the
        # optimiser steps away and the profile records "integration failed" as huge RMSE
        y = np.full((2, len(d["t"])), 1e3)
        y[0, 0], y[1, 0] = d["Te"][0], d["Tc"][0]
        return y
    return s.y


def fit(series, beta, p0):
    """Shooting fit of (a,k0,r,d,Tcool) at fixed beta over series (shared params)."""
    def resid(p):
        out = []
        for d in series:
            y = roll(d, p, beta)
            out += [y[0] - d["Te"], 4 * (y[1] - d["Tc"])]
        return np.concatenate(out)
    s = least_squares(resid, p0, diff_step=1e-3, max_nfev=80)
    rmse = [float(np.sqrt(np.mean((roll(d, s.x, beta)[0] - d["Te"]) ** 2))) for d in series]
    return s.x, rmse


t0 = time.time()
rows = []

print("=== A) per-(run,fluid) beta profiles: RMSE_Te(beta) with the other 5 params refit ===")
print("run fluid | " + " ".join(f"b={b:+.4f}" for b in sorted(BETA_GRID)) + " | beta_hat  gain%")
for r in RUNS:
    for fl in FLUIDS:
        d = D[(r, fl)]
        res, warm = {}, P0
        for b in BETA_GRID:
            p, rm = fit([d], b, warm)
            res[b] = (p, rm[0]); warm = p
        best = min(res, key=lambda b: res[b][1])
        base = res[0.0][1]
        print(f"{r}   {fl:5s} | " + " ".join(f"{res[b][1]:6.3f}" if b in res else "      ." for b in sorted(res))
              + f" | {best:+.4f}  {100 * (1 - res[best][1] / base):5.1f}", flush=True)
        for b, (p, rm) in res.items():
            rows.append(dict(part="A", fluid=fl, run=r, fold=np.nan, beta=b, rmse=rm,
                             a=float(np.exp(p[0])), k0=float(np.exp(p[1]))))

print("\n=== B) LORO transfer: fit (a,k0,r,d,Tcool)+beta on 2 runs, simulate held-out run ===")
BETA_B = [-0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03]
print("fluid held-out | beta_hat(train) | hold RMSE @b=0 / @b_hat / @best | verdict")
for fl in FLUIDS:
    for ho in RUNS:
        tr = [D[(r, fl)] for r in RUNS if r != ho]
        te = D[(ho, fl)]
        res, warm = {}, P0
        for b in BETA_B:
            p, rm_tr = fit(tr, b, warm)
            rm_ho = float(np.sqrt(np.mean((roll(te, p, b)[0] - te["Te"]) ** 2)))
            res[b] = (p, rm_tr, rm_ho); warm = p
        bhat = min(res, key=lambda b: np.mean(res[b][1]))
        bbest = min(res, key=lambda b: res[b][2])
        r0, rh, rb = res[0.0][2], res[bhat][2], res[bbest][2]
        verdict = "transfers" if rh < r0 else "does NOT transfer"
        print(f"{fl:5s} R{ho}      | {bhat:+.4f}          | {r0:.3f} / {rh:.3f} / {rb:.3f} (b={bbest:+.3f}) | {verdict}", flush=True)
        for b, (p, rtr, rho) in res.items():
            rows.append(dict(part="B", fluid=fl, run=ho, fold=ho, beta=b,
                             rmse=float(np.mean(rtr)), rmse_hold=rho,
                             a=float(np.exp(p[0])), k0=float(np.exp(p[1]))))
    print(flush=True)

pd.DataFrame(rows).to_csv("runs/outer/kappa_te_profile.csv", index=False)
print(f"DONE in {(time.time()-t0)/60:.1f} min")
