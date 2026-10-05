"""R* / DT_S invariance: with an unknown sampling interval DT_S, fitting in sample index i
instead of seconds t = DT_S * i scales all rates (a, kappa, d) by DT_S but leaves the
ratio R* = a / kappa and the predicted temperature trajectory (as a function of sample
index) unchanged. Encodes the claim in README 'Assumptions / limitations'.
Pure scipy: runs without torch/sklearn."""
import path_bootstrap
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import unittest

A, K, R, DD, TCOOL = 0.05, 0.004, 0.03, 0.003, 18.8


def rhs(t, y, Tv, s):
    a, k, d = A * s, K * s, DD * s                       # common rate scaling s (= DT_S)
    q = k * (y[0] - y[1])
    return [a * (Tv(t) - y[0]) - q, R * q - d * (y[1] - TCOOL)]


def solve(s, n=400):
    """Simulate the SAME physical experiment logged with sampling interval s seconds,
    analysed in sample-index time: rates scale by s, driver = physical ramp sampled at t = s*i."""
    ti = np.arange(n + 1) * 1.0
    Tv_spline = CubicSpline(ti, 20.0 + 60.0 * (1.0 - np.exp(-s * ti / 300.0)))
    sol = solve_ivp(rhs, [0, n], [20.0, 18.5], t_eval=ti, args=(Tv_spline, s), rtol=1e-9, atol=1e-11)
    return sol.y


class TestScalingInvariance(unittest.TestCase):
    def test_dt_s_scaling_leaves_trajectory_and_Rstar_invariant(self):
        y1 = solve(1.0)                                   # index = seconds
        y2 = solve(2.0)                                   # 2 s/sample: rates x2, driver compressed x2
        # same physical curve: y2(i) is the trajectory at t = 2i, i.e. y1 at even indices
        self.assertLess(np.abs(y2[:, :201] - y1[:, ::2]).max(), 1e-6, "index-time trajectory should match")
        # R* = a/kappa is unchanged by a common scaling by construction
        self.assertAlmostEqual((A * 2) / (K * 2), A / K, places=12)


if __name__ == "__main__":
    unittest.main()
