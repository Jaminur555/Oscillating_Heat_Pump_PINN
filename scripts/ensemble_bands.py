"""Tier 3: ensemble predictive bands from the 10 PINN seeds (loro_torch.pkl).

A) held-out rollouts: seed-mean +/- 2sd band per held-out (run, fluid); band width + empirical
   coverage of the RAW Te inside the band (tests whether seed spread reflects true error).
B) scenario bands: integrate the fold-None phys of each seed through the case-study waveforms
   -> metric CIs; does the EOHP-tracking / WOHP-ripple ranking survive seed uncertainty?
Run from repo root. Out: runs/outer/ensemble_bands.csv, log runs/outer/ensemble_bands.log.
"""
import pickle, numpy as np, pandas as pd
from ohp.data import load_all, FLUIDS, RUNS
from ohp.pinn import simulate_series
from scenario_case import ramp_hold, cyclic, slug   # waveform builders (same scenarios)

D = load_all()
res = pickle.load(open("runs/outer/loro_torch.pkl", "rb"))
SEEDS = list(range(10))
rows = []

print("=== A) held-out rollout bands (Te, 10 seeds, mean +- 2sd) ===")
print("run fluid | band_w [K] | resid [K] | cover95 | cover99.7")
for ho in RUNS:
    for fl in FLUIDS:
        d = D[(ho, fl)]
        P = np.stack([res[ho][f"PINN-const|seed{s}"]["pred"][(ho, fl)][0] for s in SEEDS])   # (10, N)
        mu, sd = P.mean(0), P.std(0)
        raw = d["raw"]["Te"]; mu_r = np.interp(d["raw"]["t"], d["t"], mu); sd_r = np.interp(d["raw"]["t"], d["t"], sd)
        w = float(np.mean(2 * sd_r)); err = np.abs(raw - mu_r)
        rows.append(dict(kind="rollout", fold=ho, fluid=fl, band_w=w, resid=float(np.sqrt(np.mean((raw - mu_r) ** 2))),
                         cover95=float(np.mean(err <= 2 * sd_r)), cover997=float(np.mean(err <= 3 * sd_r))))
        r = rows[-1]
        print(f"{ho}   {fl:5s} | {r['band_w']:8.3f}   | {r['resid']:6.3f}   | {r['cover95']:.3f}   | {r['cover997']:.3f}")

print("\n=== B) scenario bands: fold-None phys per seed (10 members) ===")
print("case        | fluid | lag_K (95% CI)      | ripple (95% CI)     | dev_mae (95% CI)")
CASES = [("A_ramp75_tau300",) + ramp_hold(75.0, 300.0) + (75.0, 1800, "A"),
         ("B_cyc75",) + cyclic(75.0) + (75.0, 3600, "B"),
         ("C_slug75",) + slug(75.0) + (75.0, 1800, "C")]
phys_seeds = [res[None][f"PINN-const|seed{s}"]["phys"] for s in SEEDS]
for name, t, Tv, Tset, i0, blk in CASES:
    d = dict(t=t, Tv=Tv, Te=np.full_like(t, Tv[0]), Tc=np.full_like(t, 18.0))   # ICs as in scenario_case
    for fl in FLUIDS:
        lag, rip, mae = [], [], []
        for ph in phys_seeds:
            te, _ = simulate_series(ph, d, fl, False)
            err = te[i0:] - Tset
            lag.append(np.mean(Tv[i0:] - te[i0:])); rip.append(np.ptp(te[i0:])); mae.append(np.mean(np.abs(err)))
        ci = lambda v: f"[{np.percentile(v, 2.5):6.2f},{np.percentile(v, 97.5):6.2f}]"
        rows += [dict(kind=f"scen_{name}", fluid=fl, metric=m, p2_5=float(np.percentile(v, 2.5)),
                      p97_5=float(np.percentile(v, 97.5))) for m, v in
                 [("lag", lag), ("ripple", rip), ("dev_mae", mae)]]
        print(f"{name:11s} | {fl:5s} | {ci(lag):19s} | {ci(rip):19s} | {ci(mae):19s}")

pd.DataFrame(rows).to_csv("runs/outer/ensemble_bands.csv", index=False)
print("\nwrote runs/outer/ensemble_bands.csv")
