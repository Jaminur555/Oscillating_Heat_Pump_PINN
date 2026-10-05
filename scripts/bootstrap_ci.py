"""Review response, item 4: uncertainty for ODE-shoot via jackknife over the 9 (run, fluid) series.

Final all-data shot + 9 leave-one-series-out refits -> jackknife pseudo-values -> sd for every
parameter, R* per fluid, and the pairwise R* differences that define the fluid ordering.
Out: runs/outer/jackknife.csv, log runs/outer/jackknife.log. ~10-25 min, CPU.
Run from repo root: KMP_DUPLICATE_LIB_OK=TRUE python scripts/bootstrap_ci.py
"""
import pickle, time, numpy as np, pandas as pd
from ohp.data import load_all, FLUIDS, RUNS
from ohp.baselines import fit_ode_shoot

D = load_all()
series = [((r, f), D[(r, f)]) for r in RUNS for f in FLUIDS]
FL = FLUIDS

def flat(phys):
    """Parameter vector: loga(3), logK(3), logr, logd, Tcool, R*(3)."""
    la, lK = np.asarray(phys["loga"]), np.asarray(phys["logK"])
    return np.r_[la, lK, phys["logr"], phys["logd"], phys["Tcool"], la - lK]

NAMES = ([f"loga_{f}" for f in FL] + [f"logK_{f}" for f in FL] +
         ["logr", "logd", "Tcool"] + [f"Rstar_{f}" for f in FL])

t0 = time.time()
theta_full = flat(fit_ode_shoot(series)[0])
print(f"full fit done in {time.time()-t0:.0f}s", flush=True)
thetas = [theta_full]
for i, drop in enumerate(series):
    sub = [s for j, s in enumerate(series) if j != i]
    thetas.append(flat(fit_ode_shoot(sub)[0]))
    print(f"leave-out {drop[0]} done {time.time()-t0:.0f}s", flush=True)

T = np.array(thetas)                                  # (10, p): row 0 = full, rows 1..9 = -i
n = len(series)
jk = n * T[0] - (n - 1) * T[1:].mean(0)               # pseudo-values (all-data target)
jk_sd = np.sqrt((n - 1) / n * ((T[1:] - T[1:].mean(0)) ** 2).sum(0))
lo_sd = T[1:].std(0, ddof=1)                          # leave-one-out spread (conservative view)
df = pd.DataFrame(dict(param=NAMES, value=T[0], jk_mean=jk, jk_sd=jk_sd, loo_sd=lo_sd))
df.to_csv("runs/outer/jackknife.csv", index=False)
pd.DataFrame(T, columns=NAMES).to_csv("runs/outer/jackknife_fits.csv", index=False)  # all 10 fits
print(df.round(4).to_string(index=False))

iR = [NAMES.index(f"Rstar_{f}") for f in FL]
print("\n=== R* jackknife (LOG scale: value +- jk_sd; linear R* = exp(value), multiplicative factor exp(sd)) ===")
for f, i in zip(FL, iR):
    print(f"{f:5s} {T[0, i]:6.2f} +- {jk_sd[i]:.2f}")
print("\n=== pairwise R* differences (value +- jk_sd; ordering robust if |value| > 2sd) ===")
for a, b in [(0, 1), (1, 2)]:
    d_ = T[0, iR[a]] - T[0, iR[b]]
    sd_ = np.sqrt((n - 1) / n * (((T[1:, iR[a]] - T[1:, iR[b]]) - (T[1:, iR[a]] - T[1:, iR[b]]).mean()) ** 2).sum())
    print(f"R*_{FL[a]} - R*_{FL[b]} = {d_:6.2f} +- {sd_:.2f}  ({d_ / sd_:.1f} sigma)")
print("\n=== 95% t-intervals (8 df, t=2.306) for log R* and pairwise log differences ===")
from scipy.stats import t as tdist
tcrit = tdist.ppf(0.975, n - 1)
for f, i in zip(FL, iR):
    print(f"log R*_{f}: [{T[0, i] - tcrit * jk_sd[i]:.2f}, {T[0, i] + tcrit * jk_sd[i]:.2f}]"
          f"  -> R* in [{np.exp(T[0, i] - tcrit * jk_sd[i]):.1f}, {np.exp(T[0, i] + tcrit * jk_sd[i]):.1f}]")
for a, b in [(0, 1), (1, 2)]:
    d_ = T[0, iR[a]] - T[0, iR[b]]
    sd_ = np.sqrt((n - 1) / n * (((T[1:, iR[a]] - T[1:, iR[b]]) - (T[1:, iR[a]] - T[1:, iR[b]]).mean()) ** 2).sum())
    print(f"log R*_{FL[a]}-log R*_{FL[b]}: [{d_ - tcrit * sd_:.2f}, {d_ + tcrit * sd_:.2f}]  (0 outside: ordering significant)")
print("DONE")
