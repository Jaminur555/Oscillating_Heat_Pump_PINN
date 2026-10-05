"""Tier 2 (8): adiabatic TCs as held-out validation for the two-node model / twin."""
import pickle, numpy as np
from ohp.data import load_all, FLUIDS, RUNS

D = load_all()
t = pickle.load(open("runs/outer/loro_torch.pkl", "rb"))

print("=== A) in-sample: Ta ~ w1*Te + w2*Tc (no intercept), per run ===")
print("run fluid | w1    w2   | RMSE_mix | RMSE_Te-only  RMSE_Tv")
for r in RUNS:
    for fl in FLUIDS:
        d = D[(r, fl)]
        X = np.c_[d["Te"], d["Tc"]]
        w, *_ = np.linalg.lstsq(X, d["Ta"], rcond=None)
        rm = np.sqrt(np.mean((X @ w - d["Ta"]) ** 2))
        rte = np.sqrt(np.mean((d["Te"] - d["Ta"]) ** 2))
        rtv = np.sqrt(np.mean((d["Tv"] - d["Ta"]) ** 2))
        print(f"{r}   {fl:5s} | {w[0]:.3f} {w[1]:.3f} | {rm:8.3f} | {rte:8.3f}    {rtv:8.3f}")

print("\n=== B) LORO: fit (w1,w2) per fluid on 2 runs, predict held-out run's Ta ===")
print("fluid | held-out RMSE_mix | RMSE_mean-predictor")
for fl in FLUIDS:
    rows = []
    for ho in RUNS:
        tr = [D[(r, fl)] for r in RUNS if r != ho]
        X = np.vstack([np.c_[d["Te"], d["Tc"]] for d in tr]); y = np.concatenate([d["Ta"] for d in tr])
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        d = D[(ho, fl)]
        rm = np.sqrt(np.mean((np.c_[d["Te"], d["Tc"]] @ w - d["Ta"]) ** 2))
        rmean = np.sqrt(np.mean((0.5 * (d["Te"] + d["Tc"]) - d["Ta"]) ** 2))
        rows.append((ho, rm, rmean))
    print(fl, " | ".join(f"R{h}: {rm:.3f} (mean {rmean:.3f})" for h, rm, rmean in rows))

print("\n=== C) twin-chain: same w, but Te/Tc from PINN-const fold predictions ===")
print("fluid | held-out RMSE_Ta: measured Te/Tc -> model Te/Tc (seed0, seed-avg)")
for fl in FLUIDS:
    out = []
    for ho in RUNS:
        tr = [D[(r, fl)] for r in RUNS if r != ho]
        X = np.vstack([np.c_[d["Te"], d["Tc"]] for d in tr]); y = np.concatenate([d["Ta"] for d in tr])
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        d = D[(ho, fl)]
        rm_m = np.sqrt(np.mean((np.c_[d["Te"], d["Tc"]] @ w - d["Ta"]) ** 2))
        rms = []
        for s in range(10):
            pe, pc = t[ho][f"PINN-const|seed{s}"]["pred"][(ho, fl)]
            rms.append(np.sqrt(np.mean((pe * w[0] + pc * w[1] - d["Ta"]) ** 2)))
        out.append(f"R{ho}: {rm_m:.3f} -> {np.mean(rms):.3f}")
    print(fl, " | ".join(out))

print("\n=== D) one-parameter mix (w2 = 1 - w1), LORO: closed-form LS on the training runs ===")
print("fluid | held-out (w1, RMSE)")
for fl in FLUIDS:
    out = []
    for ho in RUNS:
        tr = [D[(r, fl)] for r in RUNS if r != ho]
        X = np.concatenate([d["Te"] - d["Tc"] for d in tr])       # Ta-Tc = w1*(Te-Tc)
        y = np.concatenate([d["Ta"] - d["Tc"] for d in tr])
        w1 = float(X @ y / (X @ X))
        d = D[(ho, fl)]
        rm = np.sqrt(np.mean((w1 * d["Te"] + (1 - w1) * d["Tc"] - d["Ta"]) ** 2))
        out.append(f"R{ho}: w1={w1:.2f}, {rm:.3f}")
    print(fl, " | ".join(out))
