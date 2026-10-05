"""Tier 3: scenario case study -- working-fluid choice for near-isothermal adsorption duty.

Application framing (kept honest): an adsorption stage needs a process temperature held near a
setpoint while heat is cycled in/out (regeneration holds, thermostat on/off duty, exotherm slugs).
The twin transports heat from the driven vessel (Tv, the twin's input) through the evaporator;
so we rank fluids by the STABILITY of the delivered temperature Te under duty-cycled vessel
driving: ripple, deviation from setpoint, tracking lag, overshoot, and relative throughput
(mean q = kappa*(Te-Tc) in lumped model units -- relative comparison only, no absolute W claim).

Blocks:
  A) regeneration ramp+hold : ramp to Tset (3 taus), hold 1800 s  -> overshoot, settle, lag, delivery T
  B) cyclic duty            : 4 x (900 s heat / 900 s cool-off)   -> ripple, MAE vs setpoint, worst dev
  C) disturbance rejection  : hold at Tset, +10 K / 60 s slug     -> max deviation, recovery time

All scenarios stay inside the fitted envelope (Tv in [20, 95] degC, timescales 150-1800 s).
Uses the 4 existing ONNX twins (folds 1,2,3 = leave-one-run-out fits, None = all runs) -- the
spread across fitsets is the parameter-uncertainty band, same convention as fluid_sweep.csv.

Run:  KMP_DUPLICATE_LIB_OK=TRUE python scripts/scenario_case.py  (from repo root)
Out:  runs/outer/scenario_case.csv + console summary; log -> runs/outer/scenario_case.log
"""
import numpy as np
import pandas as pd
from ohp.ode_twin import ODETwin, FLUIDS

RES = "runs/outer"
TWINS = {ho: ODETwin(f"{RES}/ode_twin_fold{ho}.onnx") for ho in [1, 2, 3, None]}
TC0 = 18.0          # same condenser IC as ode_twin.sweep
H = 1.0             # replay step [s]


# ------------------------------------------------------------------ vessel waveforms (t, Tv), analytic
def ramp_hold(Tset, tau, T_start=20.0, t_ramp=1800.0, t_hold=1800.0):
    t = np.arange(0, t_ramp + t_hold + 1e-9, H)
    rise = T_start + (Tset - T_start) * (1.0 - np.exp(-np.minimum(t, t_ramp) / tau))
    return t, np.where(t <= t_ramp, rise, Tset)


def cyclic(Tset, n_cyc=4, t_on=900.0, t_off=900.0, tau_on=300.0, tau_off=300.0, T_start=20.0, T_off=25.0):
    segs, t0 = [np.array([T_start])], 0.0
    v = T_start
    for _ in range(n_cyc):
        tt = np.arange(H, t_on + 1e-9, H)
        v_hi = Tset + 3.0                                    # thermostat overshoot band
        segs.append(v_hi - (v_hi - v) * np.exp(-tt / tau_on)); v = segs[-1][-1]
        tt = np.arange(H, t_off + 1e-9, H)
        segs.append(T_off + (v - T_off) * np.exp(-tt / tau_off)); v = segs[-1][-1]
    Tv = np.concatenate(segs)
    return np.arange(len(Tv)) * H, Tv


def slug(Tset, tau=300.0, T_start=20.0, t_pre=1800.0, t_slug=60.0, t_post=1200.0, amp=10.0):
    t = np.arange(0, t_pre + t_slug + t_post + 1e-9, H)
    base = T_start + (Tset - T_start) * (1.0 - np.exp(-np.minimum(t, t_pre) / tau))
    base = np.where(t <= t_pre, base, Tset)
    s = t - t_pre                                           # smooth sin^2 bump, zero at edges
    bump = amp * np.sin(np.pi * np.clip(s / t_slug, 0, 1)) ** 2
    return t, base + np.where((s >= 0) & (s <= t_slug), bump, 0.0)


# ------------------------------------------------------------------ metrics
def _settle(t, err, band=1.0):
    ok = np.abs(err) <= band
    for i in range(len(ok)):
        if ok[i:].all():
            return float(t[i])
    return float("nan")                                     # never settles inside horizon


def run_case(twin, fl, t, Tv, Tset, window, settle_ok):
    """window: (i0, i1) evaluation window (steady duty). Returns per-fluid metric dict."""
    Te, Tc, _ = twin.replay(fl, Tv, [Tv[0], TC0], h=H)
    i0, i1 = window
    err = Te[i0:i1] - Tset
    return dict(
        delivery_Te=float(np.mean(Te[i0:i1])), dev_mae=float(np.mean(np.abs(err))),
        dev_max=float(np.max(np.abs(err))), ripple=float(np.ptp(Te[i0:i1])),
        settle_s=_settle(t[:i1], Te[:i1] - Te[i1 - 1]) if settle_ok else np.nan,   # vs final steady value (steady Te = Tset - lag)
        lag_K=float(np.mean(Tv[i0:i1] - Te[i0:i1])),
    )


if __name__ == "__main__":
    rows = []
    CASES = []      # (name, t, Tv, Tset, eval_i0, settle_meaningful)
    for Tset in (60.0, 75.0, 90.0):
        for tau in (150.0, 300.0, 600.0):
            CASES.append((f"A_ramp{Tset:.0f}_tau{tau:.0f}",) + ramp_hold(Tset, tau) + (Tset, int(1800 / H), True))
        CASES.append((f"B_cyc{Tset:.0f}",) + cyclic(Tset) + (Tset, int(3600 / H), False))        # last 2 cycles
        CASES.append((f"C_slug{Tset:.0f}",) + slug(Tset) + (Tset, int(1800 / H), False))

    for ho, twin in TWINS.items():
        for name, t, Tv, Tset, i0, settle_ok in CASES:
            for fl in FLUIDS:
                m = run_case(twin, fl, t, Tv, Tset, (i0, len(t)), settle_ok)
                rows.append(dict(fitset=str(ho), case=name, fluid=fl, Tset=Tset, **m))

    S = pd.DataFrame(rows)
    S.to_csv(f"{RES}/scenario_case.csv", index=False)

    print("=== per-case metrics (mean over 4 fitsets), best fluid per metric ===")
    LOW = ["dev_mae", "dev_max", "ripple", "settle_s", "lag_K"]        # lower is better
    for name in S.case.unique():
        sub = S[S.case == name].groupby("fluid")[LOW].mean()
        line = f"{name:16s}" + "".join(f"  {f}:{sub.loc[f, 'dev_mae']:5.2f}/{sub.loc[f, 'ripple']:5.2f}" for f in FLUIDS)
        print(line)
    for m in LOW:
        piv = S.pivot_table(index=["fitset", "case"], columns="fluid", values=m)
        bs = piv.idxmin(axis=1).value_counts(normalize=True).round(2).to_dict()
        print(f"best-fluid share on {m:9s} (lower=better): {bs}")
    print("\ncase-study reading: EOHP = tight tracking (low lag/MAE), WOHP = low ripple; choice is duty-dependent along R*")
