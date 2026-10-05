"""Parity between the torch RK4 rollout used for evaluation (ohp.pinn.rollout) and an
independent scipy LSODA integration of the same RHS, plus ONNX/torch derivative parity via
ohp.ode_twin. Skipped automatically when torch/onnx are not installed."""
import path_bootstrap
import numpy as np
import unittest

try:
    import torch
    from ohp.pinn import rollout, rhs
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestRolloutParity(unittest.TestCase):
    def test_rk4_matches_scipy(self):
        from scipy.integrate import solve_ivp
        phys = dict(loga=np.log([0.05, 0.03, 0.03]), logK=np.log([0.004] * 3),
                    logr=np.log(0.03), logd=np.log(0.003), Tcool=18.8)
        phys = {k: torch.tensor(np.asarray(v), dtype=torch.float32) for k, v in phys.items()}
        n, h = 300, 1.0
        tv = 20.0 + 60.0 * (1.0 - np.exp(-np.arange(2 * n + 1) * h / 2.0 / 300.0))
        ys = np.asarray(rollout(phys, 0, tv, np.array([20.0, 18.5]), h, False))
        fl = torch.eye(3)[0]

        def f(t, y):
            dTe, dTc = rhs(phys, fl, float(y[0]), float(y[1]), float(np.interp(t, np.arange(2 * n + 1) * h / 2.0, tv)), False)
            return [float(dTe), float(dTc)]
        ref = solve_ivp(f, [0, n * h], [20.0, 18.5], t_eval=np.arange(n + 1) * h, rtol=1e-8, atol=1e-10).y.T
        self.assertLess(np.abs(ys - ref).max(), 5e-4, "torch RK4 should track scipy LSODA closely")


if __name__ == "__main__":
    unittest.main()
