"""Baselines evaluated against the PINN.
  1. ODE-shoot : same lumped ODE (constant kappa_f), parameters fitted by shooting / nonlinear least squares
  2. MLP       : black-box network   (t, Tv, dTv/dt, fluid) -> (Te, Tc)
  3. GP        : Gaussian process    (t, Tv, dTv/dt, fluid) -> (Te, Tc)
Only (1) uses the physics. All methods receive the same measured initial state: (1) and the PINN
rollout start from it, (2),(3) get Te(0), Tc(0) as constant input features (fair-baseline fix, 2026-10).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, DotProduct
from sklearn.preprocessing import StandardScaler
from ohp.data import FLUIDS

# ----------------------------------------------------------------------------- ODE shooting
def _unpack(lp):
    p = np.exp(lp)
    return dict(a=p[0:3], K=p[3:6], r=p[6], d=p[7], Tcool=p[8])

def _roll(P, d, i):
    Tv = CubicSpline(d["t"], d["Tv"])
    def f(t, y):
        q = P["K"][i] * (y[0] - y[1])
        return [P["a"][i] * (Tv(t) - y[0]) - q, P["r"] * q - P["d"] * (y[1] - P["Tcool"])]
    return solve_ivp(f, [d["t"][0], d["t"][-1]], [d["Te"][0], d["Tc"][0]], t_eval=d["t"],
                     rtol=1e-6, atol=1e-8, method="LSODA").y

def fit_ode_shoot(series, Tc_weight=4.0):
    def resid(lp):
        P = _unpack(lp); out = []
        for (k, d) in series:
            y = _roll(P, d, FLUIDS.index(k[1]))
            out += [y[0] - d["Te"], Tc_weight * (y[1] - d["Tc"])]
        return np.concatenate(out)
    p0 = np.r_[[0.055, 0.032, 0.027], [0.0048] * 3, 0.03, 0.003, 18.8]
    # NOTE diff_step: the default (~1e-8) is swamped by integrator tolerance and stalls the optimiser
    s = least_squares(resid, np.log(p0), diff_step=1e-3, max_nfev=100)
    P = _unpack(s.x)
    phys = dict(loga=np.log(P["a"]), logK=np.log(P["K"]), logr=np.log(P["r"]), logd=np.log(P["d"]), Tcool=P["Tcool"])
    return phys, s

# ----------------------------------------------------------------------------- black-box features
def _feats(d, fluid, tscale=1800.0):
    oh = np.zeros((len(d["t"]), 3)); oh[:, FLUIDS.index(fluid)] = 1
    ic = np.full((len(d["t"]), 2), [d["Te"][0], d["Tc"][0]])   # same IC the ODE/PINN rollouts start from
    return np.c_[d["t"] / tscale, d["Tv"], d["Tv_dot"] * 100.0, oh, ic]

def _stack(series):
    X = np.vstack([_feats(d, k[1]) for k, d in series])
    Y = np.vstack([np.c_[d["Te"], d["Tc"]] for k, d in series])
    return X, Y

def fit_mlp(series, seed=0):
    X, Y = _stack(series); sx, sy = StandardScaler().fit(X), StandardScaler().fit(Y)
    m = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="tanh", solver="adam", learning_rate_init=2e-3, max_iter=3000,
                     tol=1e-7, n_iter_no_change=100, random_state=seed).fit(sx.transform(X), sy.transform(Y))
    return lambda d, fl: sy.inverse_transform(m.predict(sx.transform(_feats(d, fl)))).T

def fit_gp(series, max_pts=700):
    X, Y = _stack(series)
    idx = np.linspace(0, len(X) - 1, min(max_pts, len(X))).astype(int)      # GP is O(n^3)
    sx, sy = StandardScaler().fit(X[idx]), StandardScaler().fit(Y[idx])
    k = C(1.0) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-3, 1e4)) + C(0.1) * DotProduct() + WhiteKernel(1e-3)
    gps = [GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=5, random_state=0)
           .fit(sx.transform(X[idx]), sy.transform(Y[idx])[:, j]) for j in range(2)]
    return lambda d, fl: sy.inverse_transform(np.c_[[g.predict(sx.transform(_feats(d, fl))) for g in gps]].T).T


# ----------------------------------------------------------------------------- SINDy (pysindy, CPU)
def fit_sindy(series, threshold=0.05, degree=2):
    """Sparse identification of the two-state ODE with Tv + fluid one-hot as control input.
    States/controls scaled by std, time in hours (so coefficients and threshold are O(1e-2..1))."""
    import pysindy as ps
    X, U, T = [], [], []
    for k, d in series:
        oh = np.zeros((len(d["t"]), 3)); oh[:, FLUIDS.index(k[1])] = 1
        X.append(np.c_[d["Te"], d["Tc"]]); U.append(np.c_[d["Tv"], oh]); T.append(d["t"] / 3600.0)
    sx = np.vstack(X).std(0)
    su = np.where(np.vstack(U).std(0) > 0, np.vstack(U).std(0), 1.0)
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold),
                     feature_library=ps.PolynomialLibrary(degree=degree))
    model.fit([x / sx for x in X], t=T, u=[u / su for u in U])

    def predict(d, fl):
        oh = np.zeros((len(d["t"]), 3)); oh[:, FLUIDS.index(fl)] = 1
        t = d["t"] / 3600.0
        u = np.c_[d["Tv"], oh] / su
        ui = lambda th: np.array([np.interp(th, t, u[:, j]) for j in range(u.shape[1])])
        # own RK4 with state clipping: pysindy's solve_ivp rollout diverges on extrapolation
        # folds (rhs returns NaN) -- clipped states keep the failure finite and measurable.
        h = 1.0 / 120.0                                     # 30 s in hours
        tg = np.arange(t[0], t[-1] + h / 2, h)
        y = np.array([d["Te"][0], d["Tc"][0]]) / sx
        f = lambda yy, uu: model.predict(yy[None, :], u=uu[None, :])[0]
        ys = [y.copy()]
        for j in range(len(tg) - 1):
            u0, uh, u1 = ui(tg[j]), ui(tg[j] + h / 2), ui(tg[j] + h)
            k1 = f(y, u0); k2 = f(y + h / 2 * k1, uh); k3 = f(y + h / 2 * k2, uh); k4 = f(y + h * k3, u1)
            y = np.clip(y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), -20 / sx, 200 / sx)
            ys.append(y.copy())
        ys = np.array(ys)
        out = np.c_[np.interp(t, tg, ys[:, 0]), np.interp(t, tg, ys[:, 1])]
        return (out * sx).T
    return predict
