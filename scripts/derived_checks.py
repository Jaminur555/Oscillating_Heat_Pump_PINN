"""Checks on the derived sheets: (a) vessel-wall conduction columns, (b) Run 3 MOHP transients, (c) EOHP R*_eff drift."""
import numpy as np, pandas as pd, warnings, json
from ohp.config import RUNS_DIR as RES, FIG_DIR as FIG
warnings.filterwarnings("ignore")
out = {}
# ---- (a) vessel-wall conduction columns
k, L, ri, ro = 399.0, 0.1, 0.039, 0.04
A_ves = 0.034
for run in [1, 2, 3]:
    d = pd.read_excel(f"data_raw/HCOHP Derived Data - Run {run}.xlsx", header=None)
    dT = pd.to_numeric(d.iloc[6:, 12], errors="coerce").values
    flux = pd.to_numeric(d.iloc[6:, 16], errors="coerce").values
    pw = pd.to_numeric(d.iloc[6:, 21], errors="coerce").values
    m = np.isfinite(dT) & np.isfinite(flux) & (np.abs(dT) > 1e-9)
    ratio = flux[m] / dT[m]
    Qcyl = 2 * np.pi * k * L / np.log(ro / ri)          # W/K, true cylindrical conduction
    out[f"run{run}"] = dict(flux_per_dT_median=float(np.median(ratio)), flux_per_dT_sd=float(np.std(ratio)),
                            Qcyl_W_per_K=float(Qcyl), ratio_to_Qcyl=float(np.median(ratio) / Qcyl),
                            power_over_flux=float(np.nanmedian(pw[m] / flux[m])))
print("(a) conduction columns:", json.dumps(out, indent=1))

# ---- (b) Run 3 MOHP transients: do all thermocouples jump together?
f = "data_raw/HCOHP Primary Data - Run 3.xlsx"
res_b = {}
for fl in ["EOHP", "MOHP", "WOHP"]:
    df = pd.read_excel(f, sheet_name=f"{fl} R3", header=5).dropna(how="all")
    ev = [c for c in df.columns if "Evap" in c]; co = [c for c in df.columns if "Cond" in c]; ad = [c for c in df.columns if "Adiab" in c]
    Tv = df[[c for c in df.columns if "Vessel" in c]].mean(axis=1).values
    for tc in (400, 830):
        sl = slice(tc - 40, tc + 40)
        # largest 10 s change in each channel in the window (K)
        ch = {}
        for c in ev + ad + co:
            x = df[c].values.astype(float)[sl]; ch[c.replace(fl + " ", "")] = float(np.max(np.abs(x[10:] - x[:-10])))
        Tvx = Tv[sl]; ch["Tv(mean)"] = float(np.max(np.abs(Tvx[10:] - Tvx[:-10])))
        res_b[f"{fl}@{tc}s"] = ch
tab = pd.DataFrame(res_b).T.round(2)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\n(b) max |dT| over 10 s within +-40 s of event (K):\n", tab.to_string())
tab.to_csv(RES + "/run3_transient_check.csv")

# ---- (c) EOHP effective resistance R_eff(t) = (Te-Tc)/(Tsurf-Te) vs Te, per run, per fluid (outer-surface driver)
rows = []
for run in [1, 2, 3]:
    for fl in ["EOHP", "MOHP", "WOHP"]:
        df = pd.read_excel(f"data_raw/HCOHP Primary Data - Run {run}.xlsx", sheet_name=f"{fl} R{run}", header=5).dropna(how="all")
        Te = df[[c for c in df.columns if "Evap" in c]].mean(axis=1).rolling(31, center=True).mean().values
        Tc = df[[c for c in df.columns if "Cond" in c]].mean(axis=1).rolling(31, center=True).mean().values
        To = df[[c for c in df.columns if "Outer" in c]].mean(axis=1).rolling(31, center=True).mean().values
        gap = To - Te
        ok = gap > 1.0
        Rr = np.where(ok, (Te - Tc) / np.where(ok, gap, 1), np.nan)
        # plateau value (last 10% of run) and values binned by Te
        n = len(Te); last = slice(int(0.9 * n), n)
        row = dict(run=run, fluid=fl, plateau_Te=np.nanmean(Te[last]), plateau_gap=np.nanmean(gap[last]), plateau_R=np.nanmean(Rr[last]))
        for lo, hi in [(35, 45), (45, 55)]:
            m = ok & (Te >= lo) & (Te < hi)
            row[f"R@Te{lo}-{hi}"] = np.nanmean(Rr[m]) if m.sum() > 20 else np.nan
        rows.append(row)
R = pd.DataFrame(rows).round(2); print("\n(c) R_eff = (Te-Tc)/(Touter-Te):\n", R.to_string()); R.to_csv(RES + "/Reff_drift.csv", index=False)
json.dump(out, open(RES + "/derived_conduction_check.json", "w"), indent=1)
