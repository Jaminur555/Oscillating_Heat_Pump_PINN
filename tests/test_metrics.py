import path_bootstrap
import unittest
import numpy as np
from ohp.metrics import metrics_raw


def _series():
    # raw: 1 s axis, line with slope 0.1 + noise-free; modelling axis: every 5 s
    t_raw = np.arange(0, 101.0)
    Te_raw = 20.0 + 0.1 * t_raw
    Tc_raw = 19.0 + 0.02 * t_raw
    t_mod = t_raw[::5]
    return dict(t=t_mod, raw=dict(t=t_raw, Te=Te_raw, Tc=Tc_raw)), Te_raw, Tc_raw


class TestMetricsRaw(unittest.TestCase):
    def test_metrics_matches_manual(self):
        d, Te_raw, Tc_raw = _series()
        # perfect prediction on the coarse grid must interpolate to the exact raw curve
        pred_te = np.interp(d["t"], d["raw"]["t"], Te_raw)
        pred_tc = np.interp(d["t"], d["raw"]["t"], Tc_raw)
        m = metrics_raw(d, pred_te, pred_tc)
        self.assertLess(m["rmse_Te"], 1e-12)
        self.assertLess(m["rmse_Tc"], 1e-12)
        # constant +1 K bias -> RMSE exactly 1
        m2 = metrics_raw(d, pred_te + 1.0, pred_tc)
        self.assertAlmostEqual(m2["rmse_Te"], 1.0, places=12)
        self.assertAlmostEqual(m2["end_err_Te"], 1.0, places=12)
        self.assertAlmostEqual(m2["max_abs_Te"], 1.0, places=12)

    def test_metrics_uses_raw_not_smoothed(self):
        # prediction 2 K off the raw curve must score 2 K, even if it matched a smoothed curve
        d, Te_raw, _ = _series()
        wrong = Te_raw[::5] + 2.0
        m = metrics_raw(d, wrong, np.zeros_like(wrong))
        self.assertAlmostEqual(m["rmse_Te"], 2.0, places=12)


if __name__ == "__main__":
    unittest.main()
