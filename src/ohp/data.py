"""Data loading for the OHP PINN project.

Per (run, fluid) series we build:
    t    : time [s]            (5.00 s per sample, confirmed from the dataset article:
                                Yeboah & Darkwa, Data in Brief 33 (2020) 106505 -- Yokogawa
                                MV2000/DX2000 logger, Omega K-type TCs)
    Tv   : vessel temperature  = mean of inner+outer surface thermocouples [degC]
    Te   : evaporator temp     = mean of 3 evaporator thermocouples
    Ta   : adiabatic temp      = mean of 2 adiabatic thermocouples
    Tc   : condenser temp      = mean of 3 condenser thermocouples
"""
import os, numpy as np, pandas as pd
from scipy.signal import savgol_filter

DT_S = 5.0                      # sampling interval in seconds (confirmed, see README "Data provenance")
FLUIDS = ["EOHP", "MOHP", "WOHP"]
RUNS = [1, 2, 3]
RAW_DIR = "data_raw"              # relative to the repo root; run python -m ohp.* from there
# Vessel driver: "mean4" (all 4 vessel TCs), "outer" (2 outer TCs; matches derived sheets) or "inner".
TV_MODE = os.environ.get("OHP_TV", "outer")   # headline driver; set OHP_TV=mean4 to reproduce results_mean4/


def _load_sheet(run, fluid):
    f = os.path.join(RAW_DIR, f"HCOHP Primary Data - Run {run}.xlsx")
    df = pd.read_excel(f, sheet_name=f"{fluid} R{run}", header=5).dropna(how="all")
    cols = [c for c in df.columns if c != "Data Sampling Time"]
    ev = [c for c in cols if "Evap" in c]
    ad = [c for c in cols if "Adiabatic" in c]
    co = [c for c in cols if "Cond" in c]
    vs = [c for c in cols if "Vessel" in c]
    vs_sel = vs if TV_MODE == "mean4" else [c for c in vs if TV_MODE.capitalize() in c]
    assert len(vs_sel) in (2, 4), vs_sel
    assert len(ev) == 3 and len(ad) == 2 and len(co) == 3 and len(vs) == 4, (ev, ad, co, vs)
    out = dict(
        Te=df[ev].mean(axis=1).values.astype(float),
        Ta=df[ad].mean(axis=1).values.astype(float),
        Tc=df[co].mean(axis=1).values.astype(float),
        Tv=df[vs_sel].mean(axis=1).values.astype(float),
        Te_sd=df[ev].std(axis=1).values.astype(float),
    )
    out["t"] = np.arange(len(out["Te"])) * DT_S
    return out


def load_all(smooth_win=31, step=5):
    """Returns dict[(run, fluid)] -> dict of arrays (smoothed + subsampled).
    Tv_dot is the time-derivative of the smoothed vessel temperature [degC/s]."""
    data = {}
    for r in RUNS:
        for fl in FLUIDS:
            d = _load_sheet(r, fl)
            sm = {}
            for k in ["Te", "Ta", "Tc", "Tv"]:
                sm[k] = savgol_filter(d[k], smooth_win, 2)
            sm["Tv_dot"] = savgol_filter(d["Tv"], smooth_win, 2, deriv=1, delta=DT_S)
            sm["t"] = d["t"]
            idx = np.arange(0, len(sm["t"]), step)
            data[(r, fl)] = {k: v[idx] for k, v in sm.items()}
            data[(r, fl)]["raw"] = d
    return data


if __name__ == "__main__":
    D = load_all()
    for k, v in D.items():
        print(k, len(v["t"]), f"Tv {v['Tv'][0]:.1f}->{v['Tv'][-1]:.1f}  Te {v['Te'][0]:.1f}->{v['Te'][-1]:.1f}  "
              f"Tc {v['Tc'][0]:.1f}->{v['Tc'][-1]:.1f}  max|Tv_dot|={np.abs(v['Tv_dot']).max():.3f}")
