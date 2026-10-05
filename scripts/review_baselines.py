"""Review response, item 1: missing baselines + one-node ablation -> runs/outer/review_baselines.pkl

(a) Trivial      : Te = Tv - c_f (per-fluid mean gap, fit on training runs)
(b) Ridge-linear : linear state-space dX/dt = W.[Te, Tc, Tv, fluid-onehot, 1] fitted by ridge
                   regression on finite-difference derivatives, rolled out by RK4 from the
                   measured IC -- the linear data-driven counterpart the review asked for.
(c) One-node     : dTe/dt = a_f (Tv - Te) - d (Te - Tcool), fitted by ODE shooting (ablation
                   of the second node; Te only, no Tc prediction).
(d) Linear-SS    : unconstrained per-fluid linear state-space dX/dt = A X + g*Tv + c (8 free
                   entries per fluid, no physical sharing), fitted by output-error shooting
                   with the same optimiser as ODE-shoot -- tests whether the physical
                   parameterisation earns anything over a free linear form of the same order.

Fitters are module-level (importable by tests); the LORO loop runs under __main__.
Run from repo root:  KMP_DUPLICATE_LIB_OK=TRUE python scripts/review_baselines.py
"""
import os, pickle, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from ohp.data import load_all, FLUIDS, RUNS
from ohp.metrics import metrics_raw

OUT = "runs/outer/review_baselines.pkl"


def te_met(d, te):
    raw = d["raw"]; e = np.interp(raw["t"], d["t"], te) - raw["Te"]
    return dict(rmse_Te=float(np.sqrt(np.mean(e ** 2))), rmse_Tc=float("nan"),
                end_err_Te=float(e[-1]), max_abs_Te=float(np.abs(e).max()))


def oh(fl, n):
    o = np.zeros((n, 3)); o[:, FLUIDS.index(fl)] = 1; return o


def fit_trivial(series):
    c = {fl: float(np.mean(np.concatenate([d["Tv"] - d["Te"] for k, d in series if k[1] == fl])))
         for fl in FLUIDS}
    return lambda d, fl: (d["Tv"] - c[fl], None)


def fit_ridge(series, alpha=1e-2, h=25.0):
    X, Y = [], []
    for k, d in series:
        Z = np.c_[d["Te"], d["Tc"], d["Tv"], oh(k[1], len(d["t"])), np.ones(len(d["t"]))]
        dZ = np.gradient(Z[:, :2], d["t"], axis=0)      # d(Te,Tc)/dt only; Tv is an input
        X.append(Z); Y.append(dZ)
    X = np.vstack(X); Y = np.vstack(Y)
    mx, sx = X.mean(0), X.std(0) + 1e-9
    my, sy = Y.mean(0), Y.std(0) + 1e-9
    Xs = (X - mx) / sx; Ys = (Y - my) / sy
    W = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(X.shape[1]), Xs.T @ Ys)   # (n_feat, 3)

    def pred(d, fl):
        n = len(d["t"])
        def f(t, y):
            z = np.r_[y, np.interp(t, d["t"], d["Tv"]), *oh(fl, 1)[0], 1.0]
            return ((z - mx) / sx) @ W * sy + my
        s = solve_ivp(f, [d["t"][0], d["t"][-1]], [d["Te"][0], d["Tc"][0]],
                      t_eval=d["t"], rtol=1e-6, atol=1e-8, method="LSODA")
        return (s.y[0], s.y[1])
    return pred


def fit_onenode(series):
    def resid(lp):
        p = np.exp(lp); out = []
        for k, d in series:
            Tv = CubicSpline(d["t"], d["Tv"]); i = FLUIDS.index(k[1])
            f = lambda t, y: p[i] * (Tv(t) - y[0]) - p[3] * (y[0] - p[4])
            y = solve_ivp(f, [d["t"][0], d["t"][-1]], [d["Te"][0]], t_eval=d["t"],
                          rtol=1e-6, atol=1e-8, method="LSODA").y[0]
            out.append(y - d["Te"])
        return np.concatenate(out)
    s = least_squares(resid, np.log([0.055, 0.032, 0.027, 0.003, 18.8]), diff_step=1e-3, max_nfev=100)
    p = np.exp(s.x)

    def pred(d, fl):
        Tv = CubicSpline(d["t"], d["Tv"]); i = FLUIDS.index(fl)
        f = lambda t, y: p[i] * (Tv(t) - y[0]) - p[3] * (y[0] - p[4])
        y = solve_ivp(f, [d["t"][0], d["t"][-1]], [d["Te"][0]], t_eval=d["t"],
                      rtol=1e-6, atol=1e-8, method="LSODA").y[0]
        return (y, None)
    return pred


def fit_linss(series):
    """Per-fluid dX/dt = A X + g*Tv + c, all 8 entries free, output-error shooting."""
    preds = {}
    for fl in FLUIDS:
        sub = [(k, d) for k, d in series if k[1] == fl]
        if not sub:
            continue

        def resid(p):
            A = p[:4].reshape(2, 2); g = p[4:6]; c = p[6:8]; out = []
            for k, d in sub:
                Tv = CubicSpline(d["t"], d["Tv"])
                f = lambda t, y: np.clip(A @ y, -1e4, 1e4) + g * Tv(t) + c
                y = solve_ivp(f, [d["t"][0], d["t"][-1]], [d["Te"][0], d["Tc"][0]], t_eval=d["t"],
                              rtol=1e-6, atol=1e-8, method="LSODA").y
                out += [y[0] - d["Te"], 4 * (y[1] - d["Tc"])]
            return np.concatenate(out)
        a0, k0, r0, d0, Tc0 = 0.03, 0.0048, 0.03, 0.003, 18.8   # physical init
        p0 = np.r_[-(a0 + k0), k0, r0 * k0, -(r0 * k0 + d0), a0, 0.0, 0.0, d0 * Tc0]
        s = least_squares(resid, p0, diff_step=1e-3, max_nfev=250)
        A, g, c = s.x[:4].reshape(2, 2), s.x[4:6], s.x[6:8]

        def pred(d, fl, A=A, g=g, c=c):
            Tv = CubicSpline(d["t"], d["Tv"])
            f = lambda t, y: np.clip(A @ y, -1e4, 1e4) + g * Tv(t) + c
            y = solve_ivp(f, [d["t"][0], d["t"][-1]], [d["Te"][0], d["Tc"][0]], t_eval=d["t"],
                          rtol=1e-6, atol=1e-8, method="LSODA").y
            return (y[0], y[1])
        preds[fl] = pred
    return lambda d, fl: preds[fl](d, fl)


if __name__ == "__main__":
    D = load_all()
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    def save(): pickle.dump(res, open(OUT, "wb"))

    for ho in [1, 2, 3]:
        tr = [((r, f), D[(r, f)]) for r in RUNS for f in FLUIDS if r != ho]
        te = [((r, f), D[(r, f)]) for r in RUNS for f in FLUIDS if r == ho]
        R = res.setdefault(ho, {})

        def run(name, fit, has_tc):
            if name in R:
                return
            t0 = time.time(); pred = fit(tr)
            out = dict(pred={}, met={})
            for k, d in te:
                p = pred(d, k[1])
                out["pred"][k] = p
                out["met"][k] = metrics_raw(d, *p) if has_tc else te_met(d, p[0])
            R[name] = out; save()
            print(f"[fold {ho}] {name}: " + " ".join(f"{k[1]}:{out['met'][k]['rmse_Te']:.2f}" for k, _ in te)
                  + f"  ({time.time()-t0:.0f}s)", flush=True)

        run("Trivial", fit_trivial, False)
        run("Ridge-linear", fit_ridge, True)
        run("One-node", fit_onenode, False)
        run("Linear-SS", fit_linss, True)

    print("\n=== fold means (rmse_Te, mean over fluids) ===")
    for nm in ["Trivial", "Ridge-linear", "One-node", "Linear-SS"]:
        row = []
        for ho in [1, 2, 3]:
            v = [res[ho][nm]["met"][(ho, fl)]["rmse_Te"] for fl in FLUIDS]
            row.append(f"{np.mean(v):.2f}")
        print(f"{nm:13s}", " | ".join(row))
    print("DONE")
