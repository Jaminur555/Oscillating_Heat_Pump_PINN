"""Metrics computed against the RAW (unsmoothed, full-rate) thermocouple data.

Models are fitted on the smoothed/subsampled series (input denoising), but all REPORTED
metrics compare predictions with the raw data: predictions are linearly interpolated
from the modelling time axis onto the raw 1 s axis first.  Older results (before
2026-10) were computed against the smoothed series and are ~0.1-0.3 K more optimistic.
"""
import numpy as np


def metrics_raw(d, te, tc):
    """d: series dict with 't' (modelling axis) and 'raw' (full-rate arrays); te, tc predictions on d['t']."""
    raw = d["raw"]
    te_i = np.interp(raw["t"], d["t"], te)
    tc_i = np.interp(raw["t"], d["t"], tc)
    e, c = te_i - raw["Te"], tc_i - raw["Tc"]
    return dict(rmse_Te=float(np.sqrt(np.mean(e ** 2))), rmse_Tc=float(np.sqrt(np.mean(c ** 2))),
                end_err_Te=float(e[-1]), max_abs_Te=float(np.abs(e).max()))
