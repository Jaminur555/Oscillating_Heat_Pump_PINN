"""Review response, item 5: ONE pipeline for every summary table.

Recomputes the LORO metrics long table + fold summary from the merged result pickles
(loro.pkl = framework-free baselines, loro_torch.pkl = live torch ODE/PINN/GRU,
review_baselines.pkl = trivial/ridge/one-node) and rewrites runs/outer/loro_metrics_long.csv
and loro_summary.csv so they can never again drift from the numbers of record.
Prints a markdown table for README/paper cross-checks.  Run from repo root.
"""
import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from ohp.data import FLUIDS, RUNS

lj = pickle.load(open("runs/outer/loro.pkl", "rb"))
rt = pickle.load(open("runs/outer/loro_torch.pkl", "rb"))
rb = pickle.load(open("runs/outer/review_baselines.pkl", "rb"))
RES = {ho: {**lj.get(ho, {}), **rt.get(ho, {}), **rb.get(ho, {})} for ho in [1, 2, 3, None]}

rows = []
for ho in RUNS:
    for nm, entry in RES[ho].items():
        if "met" not in entry:
            continue
        method, _, seed = nm.partition("|")
        for k, m in entry["met"].items():
            rows.append(dict(fold=ho, run=k[0], fluid=k[1], method=method,
                             seed=int(seed[4:]) if seed else -1, **m))
L = pd.DataFrame(rows)
L.to_csv("runs/outer/loro_metrics_long.csv", index=False)

S = L.groupby(["fold", "method", "seed"])[["rmse_Te", "rmse_Tc"]].mean().reset_index()
S = S.groupby(["fold", "method"])[["rmse_Te", "rmse_Tc"]].agg(["mean", "std", "count"])
S.round(3).to_csv("runs/outer/loro_summary.csv")

print("=== fold means rmse_Te (markdown) ===")
piv = L[L.seed == -1].groupby(["method", "fold"])["rmse_Te"].mean().unstack()
pivs = L[L.seed >= 0].groupby(["method", "fold"])["rmse_Te"].agg(["mean", "std"]).round(2)
print(piv.round(2).to_markdown())
print("\nseeded (mean+-sd):")
for m in pivs.index.get_level_values(0).unique():
    r = pivs.loc[m]
    print(f"  {m:12s}", " | ".join(f"{v:.2f}+-{s:.2f}" for v, s in zip(r['mean'], r['std'])))
print("\nwrote runs/outer/loro_metrics_long.csv + loro_summary.csv")
