"""Sensitivity of ODE-shoot LORO results and R* to the choice of vessel driver Tv.
Variants: mean4 (current), outer (2 outer TCs), inner (2 inner TCs)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import ohp.data as data
from ohp.data import FLUIDS, RUNS
from ohp.baselines import fit_ode_shoot, _roll
from scipy.signal import savgol_filter

VARIANT = sys.argv[1]
orig = data._load_sheet
def patched(run, fluid):
    d = orig(run, fluid)
    f = f"{data.RAW_DIR}/HCOHP Primary Data - Run {run}.xlsx"
    df = pd.read_excel(f, sheet_name=f"{fluid} R{run}", header=5).dropna(how="all")
    key = {"outer": "Outer", "inner": "Inner"}.get(VARIANT, "")   # "" -> all four vessel TCs (mean4)
    cols = [c for c in df.columns if "Vessel" in c and key in c]
    d["Tv"] = df[cols].mean(axis=1).values.astype(float)
    return d
data._load_sheet = patched
D = data.load_all()

def rmse(phys, k, d):
    P = dict(a=np.exp(phys["loga"]), K=np.exp(phys["logK"]), r=np.exp(phys["logr"]), d=np.exp(phys["logd"]), Tcool=phys["Tcool"])
    y = _roll(P, d, FLUIDS.index(k[1])); return np.sqrt(np.mean((y[0]-d["Te"])**2))

rows = []
for ho in [1, 2, 3, None]:
    tr = [((r, f), D[(r, f)]) for r in RUNS for f in FLUIDS if r != ho]
    phys, _ = fit_ode_shoot(tr)
    Rs = np.exp(phys["loga"] - phys["logK"])
    if ho is None:
        print(VARIANT, "R* (all runs) E/M/W:", np.round(Rs, 2)); continue
    m = np.mean([rmse(phys, k, D[k]) for k in D if k[0] == ho])
    print(VARIANT, f"holdout {ho}: mean Te RMSE {m:.3f} K | R* {np.round(Rs,2)}", flush=True)
