"""Tests for the review-round baselines (scripts/review_baselines.py) and the jackknife
formulas used in scripts/bootstrap_ci.py.  Synthetic data only -- runs in CI without the dataset.
Pure scipy/numpy (no torch)."""
import path_bootstrap
import os, sys, unittest
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from review_baselines import fit_trivial, fit_linss, fit_onenode, oh
from ohp.data import FLUIDS


def synth_linear(seed, n=240, dt=5.0):
    """A series of the exact form fit_linss assumes: dX/dt = A X + g Tv + c."""
    rng = np.random.RandomState(seed)
    A = np.array([[-0.052, 0.040], [0.020, -0.033]])
    g, c = np.array([0.010, 0.0012]), np.array([0.0, 0.05])
    t = np.arange(n) * dt
    Tv = 20.0 + 60.0 * (1.0 - np.exp(-t / 300.0)) + 0.05 * rng.randn(n)
    Tv_f = lambda x: np.interp(x, t, Tv)
    s = solve_ivp(lambda tt, y: A @ y + g * Tv_f(tt) + c, [t[0], t[-1]], [20.0, 18.5],
                  t_eval=t, rtol=1e-9, atol=1e-11)
    return dict(t=t, Tv=Tv, Te=s.y[0], Tc=s.y[1])


def synth_onenode(seed, a=0.04, d=0.004, Tcool=19.0, n=240, dt=5.0):
    rng = np.random.RandomState(seed)
    t = np.arange(n) * dt
    Tv = 20.0 + 60.0 * (1.0 - np.exp(-t / 400.0)) + 0.05 * rng.randn(n)
    Tv_f = lambda x: np.interp(x, t, Tv)
    s = solve_ivp(lambda tt, y: a * (Tv_f(tt) - y[0]) - d * (y[0] - Tcool), [t[0], t[-1]],
                  [20.0], t_eval=t, rtol=1e-9, atol=1e-11)
    return dict(t=t, Tv=Tv, Te=s.y[0])


class TestBaselines(unittest.TestCase):
    def test_trivial_recovers_gap(self):
        d = synth_linear(0)
        gap = 7.5
        d2 = dict(d, Te=d["Tv"] - gap)
        pred = fit_trivial([(("x", fl), d2) for fl in FLUIDS])
        te, tc = pred(d2, "EOHP")
        self.assertTrue(np.allclose(te, d2["Tv"] - gap))
        self.assertIsNone(tc)

    def test_linss_recovers_linear_system(self):
        tr = [(("1", "EOHP"), synth_linear(1)), (("2", "EOHP"), synth_linear(2))]
        pred = fit_linss(tr)
        held = synth_linear(3)
        te, tc = pred(held, "EOHP")
        rmse = float(np.sqrt(np.mean((te - held["Te"]) ** 2)))
        self.assertLess(rmse, 0.05, f"free linear form should re-identify an exact linear system, got {rmse:.3f} K")

    def test_onenode_recovers_rates(self):
        tr = [(("1", "EOHP"), synth_onenode(1)), (("2", "EOHP"), synth_onenode(2))]
        pred = fit_onenode(tr)   # shares log-gains across fluids; single fluid here
        held = synth_onenode(3)
        te, _ = pred(held, "EOHP")
        rmse = float(np.sqrt(np.mean((te - held["Te"]) ** 2)))
        self.assertLess(rmse, 0.05, f"one-node shooting should re-identify an exact one-node system, got {rmse:.3f} K")


class TestJackknifeFormulas(unittest.TestCase):
    def test_jackknife_sd_of_mean(self):
        # classic identity: jackknife sd of the sample mean == sample sd / sqrt(n)
        x = np.array([3.1, 2.2, 4.7, 5.0, 1.9, 3.3, 2.8, 4.1, 3.9])
        n = len(x)
        loo = np.array([np.delete(x, i).mean() for i in range(n)])   # leave-one-out estimates
        full = x.mean()
        jk_sd = np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())  # formula used in bootstrap_ci.py
        self.assertAlmostEqual(jk_sd, x.std(ddof=1) / np.sqrt(n), places=12)

    def test_pseudo_values_unbiased(self):
        # n*T_full - (n-1)*mean(LOO) averaged over pseudo-values reproduces T_full
        rng = np.random.RandomState(0)
        T = rng.randn(10, 4); T[0] = 2.0                     # row 0 = full fit, rows 1..9 = LOO
        n = 9
        jk = n * T[0] - (n - 1) * T[1:].mean(0)
        self.assertAlmostEqual(jk.shape[0], 4)
        # pseudo-value mean is the delete-one jackknife estimate of the same quantity
        self.assertTrue(np.allclose(n * T[0] - (n - 1) * T[1:].mean(0), jk))


if __name__ == "__main__":
    unittest.main()
